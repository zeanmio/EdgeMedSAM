
"""
EdgeSAM evaluation 

Contents:
  1) Setup & globals
  2) Paths & dataset
  3) Predictor worker 
  4) Summary writer
  5) Main evaluate() loop
  6) CLI
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import cv2
cv2.setNumThreads(0)

import json
import glob
import time
import torch
import numpy as np
import hashlib
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from multiprocessing import get_context
from typing import List, Tuple, Dict, Any

# ========= metrics (optional fallback) =========
try:
    from utils.metrics import dice_score, hausdorff
    _HAS_UTILS_METRICS = True
except Exception:
    _HAS_UTILS_METRICS = False
    def dice_score(p255: np.ndarray, g255: np.ndarray) -> float:
        inter = float(np.logical_and(p255 > 0, g255 > 0).sum())
        s1 = float((p255 > 0).sum()); s2 = float((g255 > 0).sum())
        return (2.0 * inter / (s1 + s2)) if (s1 + s2) > 0 else 0.0
    def hausdorff(p255: np.ndarray, g255: np.ndarray) -> float:
        return float("inf")

# ==================== 1) Globals ====================

PREDICT_TIMEOUT_SEC      = 60
DATALOADER_WORKERS       = 0
PREFETCH_FACTOR          = 2
RESTART_WORKER_EVERY_N   = 2000
FLUSH_EVERY              = 500
MODEL_LOAD_MAX_WAIT_SEC  = 180

# Reproducibility / numeric stability
try:
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
except Exception:
    pass

CTX = get_context("spawn")


# ==================== 2) Utilities ====================

def resolve_under(root_dir: str, subfolder: str, rel_or_name: str) -> str:
    """Resolve a relative or absolute path under <root_dir>/<subfolder>."""
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


class GroupedDataset(Dataset):
    """Yield (image_abs, [mask_abs], [tags]) per image."""
    def __init__(self, image_to_masks: Dict[str, List[str]], dataset_root: str):
        self.dataset_root = dataset_root
        self.items = [(img_rel, mask_list) for img_rel, mask_list in image_to_masks.items()]

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


def _filter_by_shard(mapping: Dict[str, List[str]], num_shards: int, shard_id: int):
    """Deterministic image-level sharding by md5 hash."""
    if num_shards <= 1:
        return mapping
    out = {}
    for img_rel, masks in mapping.items():
        h = int(hashlib.md5(str(img_rel).encode()).hexdigest(), 16)
        if (h % num_shards) == shard_id:
            out[str(img_rel)] = masks
    return out


def tight_bbox_from_mask(mask_u8: np.ndarray) -> List[int] or None:
    """
    Return tight bbox [x1,y1,x2,y2] for a binary mask (0/255 or 0/1).
    """
    if mask_u8 is None:
        return None
    m = (mask_u8 > 0)
    if not m.any():
        return None
    ys, xs = np.where(m)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return [x1, y1, x2, y2]


# ==================== 3) Predictor worker ====================

class PredictorWorker:
    def __init__(self, checkpoint: str, model_type: str,
                 timeout: int = PREDICT_TIMEOUT_SEC):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.timeout = timeout
        self.q_in = None
        self.q_out = None
        self.p = None
        self.num_params = None
        self._start()

    def _start(self):
        """Spawn child, wait for 'ready' handshake (or time out)."""
        self.q_in = CTX.Queue(1)
        self.q_out = CTX.Queue(1)
        self.p = CTX.Process(
            target=self._worker,
            args=(self.q_in, self.q_out,
                  self.checkpoint, self.model_type)
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
    def _worker(q_in, q_out, checkpoint, model_type):
        """Child loop: import EdgeSAM, load, serve jobs."""
        import os, sys, time as _time, cv2, numpy as np, torch
        try:
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass

        # EdgeSAM import 
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        edge_sam_root = os.path.join(root_dir, "Model", "EdgeSAM")
        if edge_sam_root not in sys.path:
            sys.path.insert(0, edge_sam_root)

        try:
            from edge_sam import sam_model_registry as _reg_es
            from edge_sam import SamPredictor as _PredES
        except Exception as e:
            q_out.put(("ready", None))
            # consume jobs with error
            while True:
                job = q_in.get()
                if job is None: break
                _, _, *_ = job
                q_out.put(("batch", {"rows":[("err", (f"import_error:{e}",), "", "")], "t_first": None}))
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _reg_es[model_type](checkpoint=checkpoint).to(device).eval()
        predictor = _PredES(model)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        q_out.put(("ready", num_params))

        while True:
            job = q_in.get()
            if job is None:
                break
            image_path, mask_paths, tags, save_masks, masks_dir = job
            try:
                # ---- Read & RGB ----
                bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    q_out.put(("batch", {"rows":[("empty_img_read_fail", (0.0, float("inf")), "", "")]*len(mask_paths),
                                         "t_first": None}))
                    continue
                image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                H_img, W_img = image_rgb.shape[:2]

                # ---- set_image timing ----
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = _time.perf_counter()
                predictor.set_image(image_rgb)  # resize->norm->pad; remember input_size/original_size
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_set = _time.perf_counter() - t0

                # Retrieve encoder features (required by EdgeSAM's predict_torch)
                feats = getattr(predictor, "features", None)
                if feats is None and hasattr(predictor, "get_image_embedding"):
                    feats = predictor.get_image_embedding()

                # Gather valid GTs and tight boxes (in original image coords)
                rows: List[Tuple[str, Tuple[float,float], str, str]] = []
                inst_boxes_img, inst_gts, rows_meta = [], [], []
                had_valid_any = False

                for mpath, tag in zip(mask_paths, tags):
                    gt = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                    if gt is None:
                        rows.append(("empty_gt_read_fail", (0.0, float("inf")), mpath, tag))
                        continue
                    if gt.max() == 0:
                        rows.append(("empty_gt_all_zero", (0.0, float("inf")), mpath, tag))
                        continue
                    bbox = tight_bbox_from_mask(gt)
                    if bbox is None:
                        rows.append(("empty_gt_all_zero", (0.0, float("inf")), mpath, tag))
                        continue

                    # map bbox (GT size) -> image size
                    x1, y1, x2, y2 = bbox
                    Hgt, Wgt = gt.shape[:2]
                    sx = W_img / float(Wgt); sy = H_img / float(Hgt)
                    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
                    X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
                    X1 = max(0, min(X1, W_img)); X2 = max(0, min(X2, W_img))
                    Y1 = max(0, min(Y1, H_img)); Y2 = max(0, min(Y2, H_img))
                    if X2 < X1: X1, X2 = X2, X1
                    if Y2 < Y1: Y1, Y2 = Y2, Y1

                    gt_up = cv2.resize((gt > 0).astype(np.uint8), (W_img, H_img), interpolation=cv2.INTER_NEAREST)

                    rows_meta.append((mpath, tag))
                    inst_boxes_img.append([X1, Y1, X2, Y2])
                    inst_gts.append(gt_up.astype(np.uint8))
                    had_valid_any = True

                # ---- First latency: set_image + first decode ----
                t_decode_first = 0.0
                if had_valid_any:
                    idx0 = next((i for i, b in enumerate(inst_boxes_img) if b is not None), None)
                    if idx0 is not None:
                        box_first = torch.tensor([inst_boxes_img[idx0]], dtype=torch.float32, device=device)
                        box_first = predictor.transform.apply_boxes_torch(box_first, (H_img, W_img))
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        t0 = _time.perf_counter()
                        # EdgeSAM: single Mask => num_multimask_outputs=1
                        _m, _s, _low = predictor.predict_torch(
                            features=feats,
                            point_coords=None, point_labels=None,
                            boxes=box_first, mask_input=None,
                            num_multimask_outputs=1, return_logits=True
                        )
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        t_decode_first = _time.perf_counter() - t0
                t_first = (t_set + t_decode_first) if had_valid_any else t_set

                # ---- Decode all boxes -> low-res logits ----
                if had_valid_any and inst_boxes_img:
                    boxes_t = torch.tensor(inst_boxes_img, dtype=torch.float32, device=device)
                    boxes_t = predictor.transform.apply_boxes_torch(boxes_t, (H_img, W_img))
                    masks_t, scores_t, lowres_t = predictor.predict_torch(
                        features=feats,
                        point_coords=None, point_labels=None,
                        boxes=boxes_t, mask_input=None,
                        num_multimask_outputs=1, return_logits=True
                    )
                    # postprocess
                    input_size = getattr(predictor, "input_size", (H_img, W_img))
                    original_size = getattr(predictor, "original_size", (H_img, W_img))

                    # lowres_t: [M, 1, h_l, w_l] logits (torch)
                    masks_up = predictor.model.postprocess_masks(
                        lowres_t, input_size=input_size, original_size=original_size
                    )  # -> [M, 1, H_img, W_img] logits

                    thr = float(getattr(predictor.model, "mask_threshold", 0.0))  # sam.py default 0.0
                    masks_bin = (masks_up > thr).squeeze(1).detach().cpu().numpy().astype(np.uint8)  # [M, H, W] in {0,1}

                    # Score
                    for j, (mpath, tag_out) in enumerate(rows_meta):
                        gt_inst = inst_gts[j]
                        pred01 = masks_bin[j]

                        if pred01.sum() == 0:
                            rows.append(("empty_pred", (0.0, float("inf")), mpath, tag_out))
                            continue

                        p255 = (pred01 > 0).astype(np.uint8) * 255
                        g255 = (gt_inst > 0).astype(np.uint8) * 255
                        if p255.sum() == 0 or g255.sum() == 0:
                            dsc, hd = 0.0, float("inf")
                        else:
                            dsc = float(dice_score(p255, g255))
                            hd  = float(hausdorff(p255, g255))

                        # optional save
                        if save_masks:
                            img_name = os.path.splitext(os.path.basename(image_path))[0]
                            save_path = os.path.join(masks_dir, f"{img_name}--{tag_out}.npy")
                            try: np.save(save_path, pred01.astype(np.uint8))
                            except Exception: pass

                        rows.append(("ok", (dsc, hd), mpath, tag_out))

                q_out.put(("batch", {"rows": rows, "t_first": t_first}))
            except Exception as e:
                q_out.put(("batch", {"rows":[("err", (str(e),), "", "")]*len(mask_paths),
                                     "t_first": None}))

    def process_batch(self, image_path: str, mask_paths: List[str], tags: List[str],
                      save_masks: bool, masks_dir: str) -> Dict[str, Any]:
        """Send job + wait (with timeout). Returns payload dict."""
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
                    return {"rows":[(tag, payload, "", "")], "t_first": None}
            time.sleep(0.01)

        # Timeout: terminate and restart
        try:
            if self.p.is_alive():
                self.p.terminate(); self.p.join(timeout=1)
        except Exception:
            pass
        self._start()
        return {"rows":[("timeout", (0.0, float("inf")), "", "")] * (len(mask_paths) if isinstance(mask_paths, list) else 1),
                "t_first": None}

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


# ==================== 4) Summary writer ====================

def _summarize_from_metrics_all(metrics_all_csv, dataset_len, total_masks_all, summary_path):
    """Aggregate metrics_all.csv into a concise summary.txt (robust single-write)."""
    import numpy as np

    dscs, hds = [], []
    reason_counts = {}
    time_by_image = {}

    # ---- parse metrics_all.csv ----
    if os.path.exists(metrics_all_csv):
        with open(metrics_all_csv, "r", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
            idx_time = header.index("Time") if "Time" in header else -1
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                # metrics
                try:
                    dscs.append(float(parts[3]))
                except Exception:
                    pass
                hd_str = (parts[4].strip().lower() if len(parts) > 4 else "")
                if hd_str not in ("inf", "nan", ""):
                    try:
                        hds.append(float(hd_str))
                    except Exception:
                        pass
                # reason
                reason = parts[5].strip()
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                # timing (first occurrence per image)
                if idx_time >= 0 and len(parts) > idx_time:
                    img = parts[1].strip() if len(parts) > 1 else ""
                    tstr = parts[idx_time].strip()
                    if img and tstr:
                        try:
                            tval = float(tstr)
                            if img not in time_by_image and np.isfinite(tval):
                                time_by_image[img] = tval
                        except Exception:
                            pass

    # ---- stats ----
    avg_dsc = float(np.mean(dscs)) if dscs else None
    avg_hd  = float(np.mean(hds))  if hds  else None

    times = list(time_by_image.values())
    avg_time = float(np.mean(times)) if times else None
    thrpt = (len(times) / sum(times)) if times and sum(times) > 0 else None  # img/s

    # ---- build lines, then write once ----
    lines = []
    lines.append("=== Evaluation Summary ===")
    lines.append(f"Images (total): {dataset_len}")
    lines.append(f"Masks (total): {total_masks_all}")
    lines.append(f"[box] Average DSC: {avg_dsc:.4f}" if avg_dsc is not None else "[box] Average DSC: N/A")
    lines.append(f"[box] Average HD: {avg_hd:.2f}"  if avg_hd  is not None else "[box] Average HD: N/A")
    if avg_time is not None:
        lines.append(f"[box] Inference Time (set_image+first decode): {avg_time:.6f} sec/img")
    if thrpt is not None:
        lines.append(f"[box] Throughput: {thrpt:.2f} img/s")
    lines.append("")
    lines.append("=== Counts (all GT/masks) ===")
    ok_cnt = reason_counts.pop("ok", 0)
    lines.append(f"OK: {ok_cnt}")
    for k in sorted(reason_counts.keys()):
        lines.append(f"{k}: {reason_counts[k]}")

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ==================== 5) Main ====================

def evaluate_dataset(dataset_name: str, dataset_root: str, mapping_file: str, checkpoint: str,
                     model_type: str = "edge_sam", save_masks: bool = False,
                     num_shards: int = 1, shard_id: int = 0, out_name: str = "predictions-edgesam"):
    """Main evaluation loop over a mapping of image→[masks], per-mask only (no area filter)."""

    worker = PredictorWorker(
        checkpoint, model_type, timeout=PREDICT_TIMEOUT_SEC
    )

    # Output dirs & CSVs
    output_root = os.path.join(dataset_root, out_name)
    output_dir  = os.path.join(output_root, f"shard{shard_id}")
    save_masks = (not args.no_save) and (not getattr(args, "no_save_npy", False))
    masks_dir = None
    if save_masks:
        masks_dir = os.path.join(output_dir, "masks_npy")
        os.makedirs(masks_dir, exist_ok=True)

    metrics_csv     = os.path.join(output_dir, "metrics.csv")
    metrics_all_csv = os.path.join(output_dir, "metrics_all.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("mask,image,tag,DSC,HD,Time\n")
    if not os.path.exists(metrics_all_csv):
        with open(metrics_all_csv, "w", encoding="utf-8") as f:
            f.write("mask,image,tag,DSC,HD,reason,Time\n")

    with open(mapping_file, 'r', encoding="utf-8") as f:
        image_to_masks = json.load(f)
    image_to_masks = _filter_by_shard(image_to_masks, num_shards, shard_id)
    print(f"[INFO] Shard {shard_id}/{num_shards}: {len(image_to_masks)} images")

    dataset = GroupedDataset(image_to_masks, dataset_root)

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
        rows = payload.get("rows", [])
        t_first = payload.get("t_first", None)
        time_str = (f"{t_first:.6f}" if (t_first is not None) else "")

        # rows are per-mask
        for (status, pl, mask_path_abs, out_tag) in rows:
            if mask_path_abs:
                rel = os.path.relpath(mask_path_abs, os.path.join(dataset_root, "masks")).replace("\\", "/")
                mask_key = os.path.splitext(rel)[0]
            else:
                mask_key = f"{base}--{out_tag or 'NA'}"

            if status == "ok":
                dsc, hd = pl
                buf_ok.append(f"{mask_key},{base},{out_tag},{dsc:.6f},{hd:.6f},{time_str}\n")
                buf_all.append(f"{mask_key},{base},{out_tag},{dsc:.6f},{hd:.6f},ok,{time_str}\n")
            elif status == "empty_pred":
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,empty_pred,{time_str}\n")
            elif status == "empty_gt_read_fail":
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,empty_gt_read_fail,{time_str}\n")
            elif status == "empty_gt_all_zero":
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,empty_gt_all_zero,{time_str}\n")
            elif status == "timeout":
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,timeout,{time_str}\n")
            elif status == "empty_img_read_fail":
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,empty_img_read_fail,{time_str}\n")
            else:
                msg = pl[0] if isinstance(pl, tuple) and pl else "error"
                msg = str(msg).replace(",", " ")[:200]
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,error:{msg},{time_str}\n")

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


# ==================== 6) CLI ====================

if __name__ == "__main__":
    import argparse
    # allow_abbrev=False + parse_known_args Ignore redundant parameters
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="edge_sam")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--sample_name", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, default=None)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--out_name", type=str, default="predictions-edgesam")
    parser.add_argument("--no_save_npy", action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] edgesam_eval: ignoring extra args: {unknown}")

    sample_root = os.path.join(args.base_dir, args.sample_name)

    mapping_file = args.mapping_file
    if not mapping_file:
        cand = [p for p in glob.glob(os.path.join(sample_root, "*.json"))
                if os.path.basename(p).lower() != "bboxes.json"]
        if not cand:
            print(f"No mapping JSON found in: {sample_root}")
            raise SystemExit(1)
        mapping_file = cand[0]

    if not os.path.exists(mapping_file):
        print(f"Mapping file not found: {mapping_file}")
        raise SystemExit(1)

    evaluate_dataset(
        dataset_name=args.sample_name,
        dataset_root=sample_root,
        mapping_file=mapping_file,
        checkpoint=args.checkpoint_path,
        model_type=args.model_type,
        save_masks=not args.no_save,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        out_name=args.out_name,
    )
