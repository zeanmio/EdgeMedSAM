
"""
EfficientSAM evaluation 

Contents:
  1) Setup & globals
  2) Paths & dataset
  3) Predictor worker 
  4) Summary writer
  5) Main evaluate() loop
  6) CLI
"""

# ===================== 1) Setup & globals =====================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import cv2
cv2.setNumThreads(0)

import json
import time
import torch
import numpy as np
import hashlib
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from multiprocessing import get_context

from utils.metrics import dice_score, hausdorff

# Tunables
PREDICT_TIMEOUT_SEC      = 60
DATALOADER_WORKERS       = 0
PREFETCH_FACTOR          = 2
RESTART_WORKER_EVERY_N   = 2000
FLUSH_EVERY              = 500
MODEL_LOAD_MAX_WAIT_SEC  = 180
TARGET_SIZE              = 1024  # EfficientSAM: 1024x1024 square

try:
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass

CTX = get_context("spawn")

# ===================== 2) Paths & dataset =====================
def resolve_under(root_dir: str, subfolder: str, rel_or_name: str) -> str:
    rel = str(rel_or_name).replace("\\", "/")
    if os.path.isabs(rel) and os.path.exists(rel):
        return os.path.normpath(rel)
    p0 = os.path.normpath(os.path.join(root_dir, rel))
    if os.path.exists(p0):
        return p0
    head = subfolder.lower() + "/"
    rel_stripped = rel[len(head):] if rel.lower().startswith(head) else rel
    p1 = os.path.normpath(os.path.join(root_dir, subfolder, rel_stripped))
    if os.path.exists(p1):
        return p1
    p2 = os.path.normpath(os.path.join(root_dir, subfolder, os.path.basename(rel_stripped)))
    return p2

class GroupedESAMDataset(Dataset):
    """Yield (image_abs, [mask_abs], [tags]) per image."""
    def __init__(self, image_to_masks, dataset_root):
        self.dataset_root = dataset_root
        self.items = [(img_path, mask_list) for img_path, mask_list in image_to_masks.items()]
    def __len__(self):
        return len(self.items)
    def _resolve_image_path(self, img_rel):
        return resolve_under(self.dataset_root, "images", str(img_rel))
    def _resolve_mask_path(self, mask_rel):
        return resolve_under(self.dataset_root, "masks", str(mask_rel))
    def __getitem__(self, idx):
        image_rel, mask_rel_paths = self.items[idx]
        image_full_path = self._resolve_image_path(image_rel)
        masks, tags = [], []
        for mask_rel_path in mask_rel_paths:
            mask_path = self._resolve_mask_path(mask_rel_path)
            tag = os.path.splitext(os.path.basename(mask_path))[0].split("--")[-1]
            masks.append(mask_path)
            tags.append(tag)
        return image_full_path, masks, tags

def _filter_by_shard(mapping, num_shards, shard_id):
    if num_shards <= 1:
        return mapping
    out = {}
    for img_rel, masks in mapping.items():
        h = int(hashlib.md5(str(img_rel).encode()).hexdigest(), 16)
        if (h % num_shards) == shard_id:
            out[str(img_rel)] = masks
    return out

# ===================== 3) Predictor worker (per-image batch) =====================
class PredictorWorker:
    def __init__(self, checkpoint, model_type, timeout=PREDICT_TIMEOUT_SEC, target_size=TARGET_SIZE):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.timeout = timeout
        self.target_size = int(target_size)
        self.q_in = None
        self.q_out = None
        self.p = None
        self.num_params = None
        self._start()

    def _start(self):
        self.q_in = CTX.Queue(1)
        self.q_out = CTX.Queue(1)
        self.p = CTX.Process(
            target=self._worker,
            args=(self.q_in, self.q_out, self.checkpoint, self.model_type, self.target_size)
        )
        self.p.daemon = True
        self.p.start()
        t0 = time.time()
        while time.time() - t0 < MODEL_LOAD_MAX_WAIT_SEC:
            if not self.q_out.empty():
                tag, payload = self.q_out.get()
                if tag == "ready":
                    self.num_params = int(payload) if payload is not None else None
                    return
            time.sleep(0.02)

    @staticmethod
    def _worker(q_in, q_out, checkpoint, model_type, target_size):
        import os, sys, cv2, numpy as np, torch, time as _time
        import torch.nn.functional as F

        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Model", "EfficientSAM")))
        try:
            from efficient_sam.efficient_sam import build_efficient_sam
        except Exception as e:
            q_out.put(("ready", None))
            while True:
                job = q_in.get()
                if job is None: break
                q_out.put(("batch", {"results":[("err", (f"import_error:{e}",))]*len(job[1]),
                                     "t_first": None, "t_all": 0.0, "prompts": 0, "t_prep": 0.0}))
            return

        def pick_variant(mt_str, ckpt_path):
            mt = (str(mt_str) or "").lower().replace("_", "-")
            if mt in ("vit-s", "vits", "vit s"):
                return 384, 6
            if mt in ("vit-t", "vitt", "vit t", ""):
                return 192, 3
            try:
                sd = torch.load(ckpt_path, map_location="cpu")
                state = sd.get("model", sd)
                pe = state.get("image_encoder.pos_embed", None)
                if pe is not None and hasattr(pe, "shape"):
                    d = int(pe.shape[-1])
                    if d == 384: return 384, 6
                    if d == 192: return 192, 3
                    return d, max(1, d // 64)
            except Exception:
                pass
            return 192, 3

        def preprocess_square_cpu(image_rgb, tgt):
            H, W = image_rgb.shape[:2]
            img = cv2.resize(image_rgb, (tgt, tgt), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().div(255.0)  # [0,1]
            return t, (H, W)

        def compute_bbox_from_mask(mask_u8):
            if mask_u8 is None or mask_u8.max() == 0:
                return None
            m = (mask_u8 > 0).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            return [x, y, x + w, y + h]

        def map_bbox_to_input_square(bxyxy, gt_hw, tgt):
            (Hgt, Wgt) = gt_hw
            if bxyxy is None or Hgt <= 0 or Wgt <= 0:
                return None
            x1, y1, x2, y2 = bxyxy
            sx = tgt / float(Wgt)
            sy = tgt / float(Hgt)
            X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
            X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
            X1 = max(0, min(X1, tgt-1)); X2 = max(0, min(X2, tgt-1))
            Y1 = max(0, min(Y1, tgt-1)); Y2 = max(0, min(Y2, tgt-1))
            if X2 < X1: X1, X2 = X2, X1
            if Y2 < Y1: Y1, Y2 = Y2, Y1
            return [X1, Y1, X2, Y2]

        def select_mask(masks_1_n_k_hw, ioupred_1_n_k, j):
            kdim = masks_1_n_k_hw.shape[2]
            if kdim == 1:
                logits = masks_1_n_k_hw[0, j, 0]
            else:
                k_best = int(torch.argmax(ioupred_1_n_k[0, j]).item())
                logits = masks_1_n_k_hw[0, j, k_best]
            return (logits >= 0).to(torch.uint8).cpu().numpy()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        embed_dim, num_heads = pick_variant(model_type, checkpoint)
        model = build_efficient_sam(
            encoder_patch_embed_dim=embed_dim,
            encoder_num_heads=num_heads,
            checkpoint=checkpoint
        ).to(device).eval()

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        q_out.put(("ready", num_params))

        while True:
            job = q_in.get()
            if job is None:
                break
            image_path, mask_paths, tags, save_masks, masks_dir = job
            try:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = _time.perf_counter()
                bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    t_prep_total = _time.perf_counter() - t0
                    q_out.put(("batch", {"results":[("err", ("read_image_fail",))]*len(mask_paths),
                                         "t_first": None, "t_all": 0.0, "prompts": 0, "t_prep": t_prep_total}))
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                t_prep_total = _time.perf_counter() - t0

                img_t, (H_img, W_img) = preprocess_square_cpu(rgb, target_size)
                img_t = img_t.to(device, non_blocking=False)

                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = _time.perf_counter()
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device=="cuda" and torch.cuda.is_available())):
                    img_embed = model.get_image_embeddings(img_t)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_set = _time.perf_counter() - t0

                boxes_input, gts, valid_flags = [], [], []
                for mpath in mask_paths:
                    gt = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                    if gt is None:
                        boxes_input.append(None); gts.append(None); valid_flags.append("empty_gt_read_fail"); continue
                    if gt.max() == 0:
                        boxes_input.append(None); gts.append(gt);   valid_flags.append("empty_gt_all_zero"); continue
                    b_gt = compute_bbox_from_mask((gt>0).astype(np.uint8)*255)
                    if b_gt is None:
                        boxes_input.append(None); gts.append(gt);   valid_flags.append("empty_gt_all_zero"); continue
                    b_in = map_bbox_to_input_square(b_gt, gt.shape[:2], target_size)
                    boxes_input.append(b_in); gts.append(gt); valid_flags.append("ok")

                results = [None]*len(mask_paths)
                t_first = None
                t_all = 0.0
                prompts = 0

                valid_idx = [i for i,flag in enumerate(valid_flags) if flag=="ok" and boxes_input[i] is not None]
                if valid_idx:
                    i0 = valid_idx[0]
                    x1,y1,x2,y2 = boxes_input[i0]
                    pts0 = np.array([[x1,y1],[x2,y2]], dtype=np.float32).reshape(1,1,2,2)
                    lbl0 = np.array([2,3], dtype=np.int64).reshape(1,1,2)
                    pts0_t = torch.from_numpy(pts0).to(device)
                    lbl0_t = torch.from_numpy(lbl0).to(device)

                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t0 = _time.perf_counter()
                    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device=="cuda" and torch.cuda.is_available())):
                        masks_1_1_k, iou_1_1_k = model.predict_masks(
                            image_embeddings=img_embed,
                            batched_points=pts0_t,
                            batched_point_labels=lbl0_t,
                            multimask_output=True,
                            input_h=target_size, input_w=target_size,
                            output_h=target_size, output_w=target_size,
                        )
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    dt_first = _time.perf_counter() - t0
                    t_first = t_set + dt_first
                    t_all += dt_first
                    prompts += 1

                    pred0 = select_mask(masks_1_1_k, iou_1_1_k, 0)
                    gt0 = gts[i0]
                    if pred0.shape != gt0.shape:
                        pred0 = cv2.resize(pred0, (gt0.shape[1], gt0.shape[0]), interpolation=cv2.INTER_NEAREST)
                    p255 = (pred0>0).astype(np.uint8)*255
                    g255 = (gt0  >0).astype(np.uint8)*255
                    if p255.sum()==0 or g255.sum()==0:
                        results[i0] = ("empty_pred", (0.0, float("inf")))
                    else:
                        dsc = float(dice_score(p255, g255))
                        hd  = float(hausdorff(p255, g255))
                        results[i0] = ("ok", (dsc, hd))

                rest_idx = [i for i in valid_idx if results[i] is None]
                if rest_idx:
                    pts_list, lbl_list = [], []
                    for i in rest_idx:
                        x1,y1,x2,y2 = boxes_input[i]
                        pts = np.array([[x1,y1],[x2,y2]], dtype=np.float32)
                        lbl = np.array([2,3], dtype=np.int64)
                        pts_list.append(pts)
                        lbl_list.append(lbl)
                    pts_all = np.stack(pts_list, axis=0).reshape(1, len(rest_idx), 2, 2)
                    lbl_all = np.stack(lbl_list, axis=0).reshape(1, len(rest_idx), 2)
                    pts_t = torch.from_numpy(pts_all).to(device)
                    lbl_t = torch.from_numpy(lbl_all).to(device)

                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t0 = _time.perf_counter()
                    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device=="cuda" and torch.cuda.is_available())):
                        masks_1_n_k, iou_1_n_k = model.predict_masks(
                            image_embeddings=img_embed,
                            batched_points=pts_t,
                            batched_point_labels=lbl_t,
                            multimask_output=True,
                            input_h=target_size, input_w=target_size,
                            output_h=target_size, output_w=target_size,
                        )
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    dt_rest = _time.perf_counter() - t0
                    t_all += dt_rest
                    prompts += len(rest_idx)

                    for j, ii in enumerate(rest_idx):
                        predj = select_mask(masks_1_n_k, iou_1_n_k, j)
                        gtj = gts[ii]
                        if predj.shape != gtj.shape:
                            predj = cv2.resize(predj, (gtj.shape[1], gtj.shape[0]), interpolation=cv2.INTER_NEAREST)
                        p255 = (predj>0).astype(np.uint8)*255
                        g255 = (gtj >0).astype(np.uint8)*255
                        if p255.sum()==0 or g255.sum()==0:
                            results[ii] = ("empty_pred", (0.0, float("inf")))
                        else:
                            dsc = float(dice_score(p255, g255))
                            hd  = float(hausdorff(p255, g255))
                            results[ii] = ("ok", (dsc, hd))

                for i, flag in enumerate(valid_flags):
                    if results[i] is not None:
                        continue
                    if flag == "empty_gt_read_fail":
                        results[i] = ("empty_gt_read_fail", (0.0, float("inf")))
                    elif flag == "empty_gt_all_zero":
                        results[i] = ("empty_gt_all_zero", (0.0, float("inf")))
                    else:
                        results[i] = ("err", ("unknown_state",))

                q_out.put(("batch", {"results": results, "t_first": t_first, "t_all": t_all,
                                     "prompts": prompts, "t_prep": t_prep_total}))
            except Exception as e:
                q_out.put(("batch", {"results":[("err", (str(e),))]*len(mask_paths),
                                     "t_first": None, "t_all": 0.0, "prompts": 0, "t_prep": 0.0}))

    def process_batch(self, image_path, mask_paths, tags, save_masks, masks_dir):
        if self.p is None or not self.p.is_alive():
            self._start()
        while not self.q_out.empty():
            try: self.q_out.get_nowait()
            except Exception: break

        self.q_in.put((image_path, mask_paths, tags, save_masks, masks_dir))
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            if not self.q_out.empty():
                tag, payload = self.q_out.get()
                if tag == "batch":
                    return payload
                else:
                    return {"results":[(tag, payload)], "t_first": None, "t_all": 0.0, "prompts": 0, "t_prep": 0.0}

            time.sleep(0.01)

        try:
            if self.p.is_alive():
                self.p.terminate(); self.p.join(timeout=1)
        except Exception:
            pass
        self._start()
        return {"results":[("timeout", (0.0, float("inf")))] * (len(mask_paths) if isinstance(mask_paths, list) else 1),
                "t_first": None, "t_all": 0.0, "prompts": 0, "t_prep": 0.0}

    def restart(self):
        try:
            if self.p and self.p.is_alive():
                self.q_in.put(None); self.p.join(timeout=1)
        except Exception:
            pass
        self._start()

    def close(self):
        try:
            if self.p and self.p.is_alive():
                self.q_in.put(None); self.p.join(timeout=1)
        except Exception:
            pass

# ===================== 4) Summary writer =====================
def _summarize_from_metrics_all(metrics_all_csv, dataset_len, total_masks_all, summary_path):
    dscs, hds = [], []
    reason_counts = {}
    time_by_image = {}
    prep_by_image = {}

    if os.path.exists(metrics_all_csv):
        with open(metrics_all_csv, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            idx_time = header.index("Time") if "Time" in header else -1
            idx_prep = header.index("Prep") if "Prep" in header else -1
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                try:
                    d = float(parts[3]); dscs.append(d)
                except Exception:
                    pass
                hd_str = parts[4].strip().lower()
                if hd_str not in ("inf", "nan", ""):
                    try: hds.append(float(hd_str))
                    except Exception: pass
                reason = parts[5].strip() if len(parts) > 5 else ""
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                img = parts[1].strip() if len(parts) > 1 else ""
                if img:
                    if idx_time >= 0 and len(parts) > idx_time:
                        tstr = parts[idx_time].strip()
                        if tstr:
                            try:
                                tval = float(tstr)
                                if img not in time_by_image and np.isfinite(tval):
                                    time_by_image[img] = tval
                            except Exception:
                                pass
                    if idx_prep >= 0 and len(parts) > idx_prep:
                        pstr = parts[idx_prep].strip()
                        if pstr:
                            try:
                                pval = float(pstr)
                                if img not in prep_by_image and np.isfinite(pval):
                                    prep_by_image[img] = pval
                            except Exception:
                                pass

    def safe_mean(arr):
        return (float(np.mean(arr)) if len(arr) else None)
    def pct(arr, p):
        if not arr:
            return None
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return None
        try:
            return float(np.percentile(a, p, method="linear"))
        except TypeError:
            return float(np.percentile(a, p, interpolation="linear"))

    avg_dsc = safe_mean(dscs)
    avg_hd  = safe_mean(hds)

    times  = list(time_by_image.values())
    preps  = list(prep_by_image.values())

    avg_time = safe_mean(times)
    avg_prep = safe_mean(preps)

    p_time = {q: pct(times, q) for q in (50,90,95,99)} if times else {}
    p_prep = {q: pct(preps, q) for q in (50,90,95,99)} if preps else {}

    thrpt_model = (len(times) / sum(times)) if times and sum(times) > 0 else None

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Summary ===\n")
        f.write(f"Images (total): {dataset_len}\n")
        f.write(f"Masks (total): {total_masks_all}\n")
        f.write(f"[box] Average DSC: {avg_dsc:.4f}\n" if avg_dsc is not None else "[box] Average DSC: N/A\n")
        f.write(f"[box] Average HD: {avg_hd:.2f}\n"  if avg_hd  is not None else "[box] Average HD: N/A\n")

        if avg_time is not None:
            f.write(f"[box] Inference Time (set_embed+first decode): {avg_time:.6f} sec/img\n")
            if thrpt_model is not None:
                f.write(f"[box] Throughput (Model-only): {thrpt_model:.2f} img/s\n")
            if p_time:
                f.write("[box] Inference Latency p50/p90/p95/p99: " + "/".join(f"{p_time[q]:.6f}" for q in (50,90,95,99)) + " sec\n")

        if avg_prep is not None:
            f.write(f"[box] Prep Time: {avg_prep:.6f} sec/img\n")
            if p_prep:
                f.write("[box] Prep Latency p50/p90/p95/p99: " + "/".join(f"{p_prep[q]:.6f}" for q in (50,90,95,99)) + " sec\n")

        f.write("\n=== Counts (all GT) ===\n")
        ok_cnt = reason_counts.pop("ok", 0)
        f.write(f"OK: {ok_cnt}\n")
        for k in sorted(reason_counts.keys()):
            f.write(f"{k}: {reason_counts[k]}\n")

# ===================== 5) Main evaluate() loop =====================
def evaluate_dataset(dataset_name, dataset_root, mapping_file, checkpoint,
                     model_type="vit_t", save_masks=False,
                     num_shards=1, shard_id=0, target_size=TARGET_SIZE,
                     out_name: str = None):
    if not mapping_file or (not os.path.exists(mapping_file)):
        raise RuntimeError(f"mapping_file must exist: {mapping_file}")
    if not out_name or not str(out_name).strip():
        raise ValueError("out_name must be provided (non-empty).")

    worker = PredictorWorker(checkpoint, model_type, timeout=PREDICT_TIMEOUT_SEC, target_size=target_size)

    output_root = os.path.join(dataset_root, str(out_name).strip())
    output_dir  = os.path.join(output_root, f"shard{shard_id}")
    masks_dir   = os.path.join(output_dir, "masks_npy")
    os.makedirs(masks_dir, exist_ok=True)

    metrics_csv = os.path.join(output_dir, "metrics.csv")
    metrics_all_csv = os.path.join(output_dir, "metrics_all.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("mask,image,tag,DSC,HD,Time,Prep\n")
    if not os.path.exists(metrics_all_csv):
        with open(metrics_all_csv, "w", encoding="utf-8") as f:
            f.write("mask,image,tag,DSC,HD,reason,Time,Prep\n")

    with open(mapping_file, 'r', encoding="utf-8") as f:
        image_to_masks = json.load(f)
    image_to_masks = _filter_by_shard(image_to_masks, num_shards, shard_id)
    dataset = GroupedESAMDataset(image_to_masks, dataset_root)

    buf_ok, buf_all = [], []

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=DATALOADER_WORKERS,
        persistent_workers=(DATALOADER_WORKERS > 0),
        prefetch_factor=(PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None),
        collate_fn=lambda xs: xs[0]
    )

    processed_masks = 0
    for image_path, mask_paths, tags in tqdm(
        loader, desc="Evaluating", dynamic_ncols=True,
        mininterval=0.5, ascii=True,
        bar_format="{desc}: |{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ):
        base = os.path.splitext(os.path.basename(image_path))[0]

        payload = worker.process_batch(image_path, mask_paths, tags, save_masks, masks_dir)
        results = payload.get("results", [])
        t_first = payload.get("t_first", None)
        t_prep  = payload.get("t_prep",  None)
        time_str = (f"{t_first:.6f}" if (t_first is not None) else "")
        prep_str = (f"{t_prep:.6f}"  if (t_prep  is not None) else "")

        for (status, pl), mask_path, tag in zip(results, mask_paths, tags):
            rel = os.path.relpath(mask_path, os.path.join(dataset_root, "masks")).replace("\\", "/")
            mask_key = os.path.splitext(rel)[0]

            if status == "ok":
                dsc, hd = pl
                buf_ok.append(f"{mask_key},{base},{tag},{dsc:.6f},{hd:.6f},{time_str},{prep_str}\n")
                buf_all.append(f"{mask_key},{base},{tag},{dsc:.6f},{hd:.6f},ok,{time_str},{prep_str}\n")
            elif status == "empty_pred":
                buf_all.append(f"{mask_key},{base},{tag},0.000000,inf,empty_pred,{time_str},{prep_str}\n")
            elif status == "empty_gt_read_fail":
                buf_all.append(f"{mask_key},{base},{tag},0.000000,inf,empty_gt_read_fail,{time_str},{prep_str}\n")
            elif status == "empty_gt_all_zero":
                buf_all.append(f"{mask_key},{base},{tag},0.000000,inf,empty_gt_all_zero,{time_str},{prep_str}\n")
            elif status == "timeout":
                buf_all.append(f"{mask_key},{base},{tag},0.000000,inf,timeout,{time_str},{prep_str}\n")
            else:
                msg = pl[0] if isinstance(pl, tuple) and pl else "error"
                msg = str(msg).replace(",", " ")[:200]
                buf_all.append(f"{mask_key},{base},{tag},0.000000,inf,error:{msg},{time_str},{prep_str}\n")

        processed_masks += len(mask_paths)

        if RESTART_WORKER_EVERY_N and processed_masks and (processed_masks % RESTART_WORKER_EVERY_N == 0):
            worker.restart()

        if len(buf_ok) >= FLUSH_EVERY:
            with open(metrics_csv, "a", encoding="utf-8") as f: f.writelines(buf_ok)
            buf_ok.clear()
        if len(buf_all) >= FLUSH_EVERY:
            with open(metrics_all_csv, "a", encoding="utf-8") as f: f.writelines(buf_all)
            buf_all.clear()

    if buf_ok:
        with open(metrics_csv, "a", encoding="utf-8") as f: f.writelines(buf_ok)
    if buf_all:
        with open(metrics_all_csv, "a", encoding="utf-8") as f: f.writelines(buf_all)

    total_images_all = len(dataset)
    total_masks_all = sum(len(v) for v in image_to_masks.values())
    summary_path = os.path.join(output_dir, "summary.txt")
    _summarize_from_metrics_all(metrics_all_csv, total_images_all, total_masks_all, summary_path)

    try:
        with open(os.path.join(output_dir, "done.flag"), "w", encoding="utf-8") as _f:
            _f.write("ok")
    except Exception:
        pass

    worker.close()

# ===================== 6) CLI =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="vit_t")  # vit_t / vit_s
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--sample_name", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--target_size", type=int, default=TARGET_SIZE)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] efficientsam.py: ignoring extra args: {unknown}")
        
    TARGET_SIZE = int(args.target_size) if args.target_size else 1024

    sample_root = os.path.join(args.base_dir, args.sample_name)
    if not args.mapping_file or (not os.path.exists(args.mapping_file)):
        print(f"Mapping file not found: {args.mapping_file}")
        raise SystemExit(1)

    evaluate_dataset(
        dataset_name=args.sample_name,
        dataset_root=sample_root,
        mapping_file=args.mapping_file,
        checkpoint=args.checkpoint_path,
        model_type=args.model_type,
        save_masks=not args.no_save,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        target_size=TARGET_SIZE,
        out_name=args.out_name,
    )
