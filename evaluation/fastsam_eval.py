
"""
FastSAM evaluation

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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics.yolo.utils.checks")

import cv2
cv2.setNumThreads(0)

import json
import time
import numpy as np
import hashlib
from tqdm import tqdm
from typing import Dict, List, Tuple, Iterable
from torch.utils.data import Dataset, DataLoader
from multiprocessing import get_context
import torch

try:
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass

from utils.metrics import dice_score, hausdorff

# Defaults
PREDICT_TIMEOUT_SEC     = 60
DATALOADER_WORKERS      = 0
PREFETCH_FACTOR         = 2
RESTART_WORKER_EVERY_N  = 2000
FLUSH_EVERY             = 500
MODEL_LOAD_MAX_WAIT_SEC = 180
DEFAULT_IMGSZ           = 640
DEFAULT_CONF            = 0.10
DEFAULT_IOU             = 0.70
DEFAULT_RETINA_MASKS    = False

CTX = get_context("spawn")


# ===================== 2) Safe resize + utilities =====================
def safe_resize(arr, w, h, interpolation=cv2.INTER_NEAREST):
    if arr is None:
        return None
    try:
        h0, w0 = arr.shape[:2]
    except Exception:
        return None
    w = int(w); h = int(h)
    if w <= 0 or h <= 0 or w0 <= 0 or h0 <= 0:
        return None
    if (w, h) == (w0, h0):
        return arr
    return cv2.resize(arr, (w, h), interpolation=interpolation)


def resolve_under(root_dir: str, subfolder: str, rel_or_name: str) -> str:
    """Resolve path against <root_dir>/<subfolder>, accepting abs/root-rel/<subfolder>/name/basename."""
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
    return os.path.normpath(os.path.join(root_dir, subfolder, os.path.basename(rel_stripped)))


def compute_bbox_from_mask(mask_u8: np.ndarray):
    """Largest external contour bbox from a binary mask; return xyxy or None."""
    if mask_u8 is None or mask_u8.max() == 0:
        return None
    m = (mask_u8 > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return [x, y, x + w, y + h]


def map_bbox_mask2img(bbox_m, img_hw: Tuple[int, int], mask_hw: Tuple[int, int]):
    """Map bbox from GT mask size to original image size."""
    if bbox_m is None:
        return None
    H_img, W_img = img_hw
    H_m, W_m = mask_hw
    sx = W_img / float(W_m); sy = H_img / float(H_m)
    x1,y1,x2,y2 = bbox_m
    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
    X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
    X1 = max(0, min(X1, W_img-1)); X2 = max(0, min(X2, W_img-1))
    Y1 = max(0, min(Y1, H_img-1)); Y2 = max(0, min(Y2, H_img-1))
    if X2 < X1: X1, X2 = X2, X1
    if Y2 < Y1: Y1, Y2 = Y2, Y1
    return [X1,Y1,X2,Y2]


def normalize_box(b, H, W, min_wh=2):
    """Clamp, order, and enforce minimum bbox size."""
    if b is None:
        return None
    x1, y1, x2, y2 = map(int, b)
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if (x2-x1+1) < min_wh:
        cx = (x1+x2)//2
        x1 = max(0, cx-(min_wh//2)); x2 = min(W-1, x1+min_wh-1)
    if (y2-y1+1) < min_wh:
        cy = (y1+y2)//2
        y1 = max(0, cy-(min_wh//2)); y2 = min(H-1, y1+min_wh-1)
    if x2 <= x1:
        x2 = min(W-1, x1+1)
    if y2 <= y1:
        y2 = min(H-1, y1+1)
    return [int(x1),int(y1),int(x2),int(y2)]


def mask_iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Binary IoU between two masks (allowing channel dims)."""
    if a.ndim == 3: a = np.any(a, axis=-1)
    if b.ndim == 3: b = np.any(b, axis=-1)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _filter_by_shard(mapping: Dict, num_shards: int, shard_id: int) -> Dict:
    """Deterministic hash-based sharding by image key."""
    if num_shards <= 1:
        return mapping
    out = {}
    for img_rel, masks in mapping.items():
        h = int(hashlib.md5(str(img_rel).encode()).hexdigest(), 16)
        if (h % num_shards) == shard_id:
            out[str(img_rel)] = masks
    return out


# ===================== 3) Dataset =====================
class GroupedDataset(Dataset):
    """Yield (image_abs, [mask_abs], [tags]) per image."""
    def __init__(self, image_to_masks: Dict, dataset_root: str):
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
            masks.append(mask_path); tags.append(tag)
        return image_full_path, masks, tags


# ===================== 4) FastSAM import shim =====================
def _import_fastsam():
    try:
        from fastsam import FastSAM as _M, FastSAMPrompt as _P
        return _M, _P, "pypi.fastsam"
    except Exception:
        pass
    try:
        from ultralytics import FastSAM as _M
        from ultralytics.models.fastsam import FastSAMPrompt as _P
        return _M, _P, "ultralytics"
    except Exception as e:
        return None, None, f"import_error:{e}"


# ===================== 5) Predictor worker =====================
class PredictorWorker:
    def __init__(self, checkpoint, imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF, iou=DEFAULT_IOU,
                 retina_masks=DEFAULT_RETINA_MASKS, timeout=PREDICT_TIMEOUT_SEC):
        self.checkpoint = checkpoint
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.retina_masks = bool(retina_masks)
        self.timeout = timeout
        self.q_in = None; self.q_out = None; self.p = None
        self.num_params = None
        self._start()

    def _start(self):
        self.q_in = CTX.Queue(1); self.q_out = CTX.Queue(1)
        self.p = CTX.Process(
            target=self._worker,
            args=(self.q_in, self.q_out, self.checkpoint, self.imgsz, self.conf, self.iou, self.retina_masks)
        )
        self.p.daemon = True; self.p.start()
        t0 = time.time()
        while time.time() - t0 < MODEL_LOAD_MAX_WAIT_SEC:
            if not self.q_out.empty():
                tag, payload = self.q_out.get()
                if tag == "ready":
                    self.num_params = int(payload) if payload is not None else None
                    return
            time.sleep(0.02)

    @staticmethod
    def _worker(q_in, q_out, checkpoint, imgsz, conf, iou, retina_masks):
        import cv2 as _cv2, numpy as _np, time as _time, torch as _torch
        _FastSAM, _FastSAMPrompt, _src = _import_fastsam()
        device = "cuda" if _torch.cuda.is_available() else "cpu"

        if _FastSAM is None:
            q_out.put(("ready", None))
            while True:
                job = q_in.get()
                if job is None: break
                _, mask_paths, *_ = job
                q_out.put(("batch", {"rows":[("err", (_src,), "", "")]*len(mask_paths), "t_first": None}))
            return

        model = _FastSAM(checkpoint)
        try: model.to(device)
        except Exception: pass
        try: (getattr(model,"model",model)).float()
        except Exception: pass
        try:
            num_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        except Exception:
            try:
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            except Exception:
                num_params = 0

        q_out.put(("ready", int(num_params)))

        def _extract_masks_tensor(results):
            r = results[0] if isinstance(results, (list,tuple)) else results
            try:
                ms = getattr(r,"masks",None)
                if ms is not None and getattr(ms,"data",None) is not None:
                    return ms.data
            except Exception:
                return None
            return None

        def _enumerate_masks_from_anns(anns, H, W, x1,y1,x2,y2) -> Iterable[_np.ndarray]:
            """Yield boolean masks cropped to bbox from various annotation shapes."""
            def _single_to_mask(a):
                if isinstance(a, _np.ndarray):
                    A = (a > 0).astype(_np.uint8)
                    if A.ndim == 2:
                        m = safe_resize(A, W, H, _cv2.INTER_NEAREST); return None if m is None else m
                    elif A.ndim == 3:
                        masks = []
                        if A.shape[1] == H and A.shape[2] == W:
                            for k in range(A.shape[0]):
                                mk = safe_resize(A[k], W, H, _cv2.INTER_NEAREST)
                                if mk is not None: masks.append(mk)
                            return masks
                        if A.shape[0] == H and A.shape[1] == W:
                            for k in range(A.shape[2]):
                                mk = safe_resize(A[:,:,k], W, H, _cv2.INTER_NEAREST)
                                if mk is not None: masks.append(mk)
                            return masks
                        B = _np.squeeze(A)
                        if B.ndim == 2:
                            m = safe_resize(B, W, H, _cv2.INTER_NEAREST); return None if m is None else m
                        if B.ndim == 3 and B.shape[1] == H and B.shape[2] == W:
                            return [safe_resize(B[k], W, H, _cv2.INTER_NEAREST) for k in range(B.shape[0])]
                        return None
                elif isinstance(a, dict) and ("segmentation" in a):
                    m = _np.zeros((H,W), dtype=_np.uint8)
                    try:
                        pts = _np.array(a["segmentation"], dtype=_np.int32).reshape(-1,2)
                        _cv2.fillPoly(m, [pts], 1)
                    except Exception:
                        return None
                    return m
                else:
                    mobj = getattr(a, "mask", None)
                    if mobj is None: return None
                    m = (_np.array(mobj) > 0).astype(_np.uint8)
                    m = safe_resize(m, W, H, _cv2.INTER_NEAREST)
                    return None if m is None else m

            seq = []
            if isinstance(anns, (list,tuple)): seq = list(anns)
            elif isinstance(anns, _np.ndarray): seq = [anns]
            elif anns is not None: seq = [anns]

            for a in seq:
                out = _single_to_mask(a)
                if out is None: continue
                arrs = out if isinstance(out, list) else [out]
                for m in arrs:
                    if m is None: continue
                    m[:y1,:] = 0; m[y2+1:,:] = 0; m[:,:x1] = 0; m[:,x2+1:] = 0
                    yield m.astype(bool)

        while True:
            job = q_in.get()
            if job is None:
                break

            image_path, mask_paths, tags, save_masks, masks_dir = job
            try:
                # ---- Forward ----
                if _torch.cuda.is_available(): _torch.cuda.synchronize()
                t0 = _time.perf_counter()
                bgr = _cv2.imread(image_path, _cv2.IMREAD_COLOR)
                if bgr is None or bgr.size == 0:
                    q_out.put(("batch", {"rows":[("err", ("read_image_fail",), "", "")]*len(mask_paths), "t_first": None}))
                    continue
                H, W = bgr.shape[:2]
                if int(H) <= 0 or int(W) <= 0:
                    q_out.put(("batch", {"rows":[("err", ("invalid_image_size",), "", "")]*len(mask_paths), "t_first": None}))
                    continue

                # model forward + NMS
                if _torch.cuda.is_available(): _torch.cuda.synchronize()
                t_forward0 = _time.perf_counter()
                try:
                    results = model(bgr, imgsz=imgsz, device=device, retina_masks=retina_masks,
                                    conf=conf, iou=iou, verbose=False, half=False)
                except TypeError:
                    results = model(bgr, imgsz=imgsz, device=device, retina_masks=retina_masks,
                                    conf=conf, iou=iou, verbose=False)
                if _torch.cuda.is_available(): _torch.cuda.synchronize()
                t_forward = _time.perf_counter() - t_forward0

                # ---- Promptor（box） ----
                _FastSAM, _FastSAMPrompt, _ = _import_fastsam()
                try:
                    promptor = _FastSAMPrompt(bgr, results, device=device)
                except Exception:
                    promptor = None

                rows = []
                first_sel_time = 0.0; first_sel_done = False
                masks_tensor = _extract_masks_tensor(results)

                for mp, tg in zip(mask_paths, tags):
                    gt = _cv2.imread(mp, _cv2.IMREAD_GRAYSCALE)
                    if gt is None:
                        rows.append(("empty_gt_read_fail", (0.0, float("inf")), mp, tg)); continue
                    if gt.max() == 0:
                        rows.append(("empty_gt_all_zero", (0.0, float("inf")), mp, tg)); continue

                    bbox_m = compute_bbox_from_mask(((gt>0).astype(_np.uint8)*255))
                    if bbox_m is None:
                        rows.append(("empty_gt_all_zero", (0.0, float("inf")), mp, tg)); continue

                    bbox_img = map_bbox_mask2img(bbox_m, (H,W), gt.shape[:2])
                    bbox_img = normalize_box(bbox_img, H, W, min_wh=2)
                    if bbox_img is None:
                        rows.append(("err", ("invalid_bbox",), mp, tg)); continue
                    x1,y1,x2,y2 = bbox_img

                    # upsample GT to image size
                    gt_up_u8 = safe_resize((gt>0).astype(_np.uint8), W, H, _cv2.INTER_NEAREST)
                    if gt_up_u8 is None:
                        rows.append(("err", ("invalid_resize_gt",), mp, tg)); continue
                    gt_bool = gt_up_u8.astype(bool)

                    pred_bool = None
                    # ----- Box prompt path (first time occurance) -----
                    if promptor is not None:
                        if _torch.cuda.is_available(): _torch.cuda.synchronize()
                        s0 = _time.perf_counter()
                        try:
                            try:
                                anns = promptor.box_prompt(bboxes=[[x1,y1,x2,y2]])
                            except TypeError:
                                anns = promptor.box_prompt(bbox=[x1,y1,x2,y2])
                        except _cv2.error:
                            anns = None
                        except Exception:
                            anns = None
                        if _torch.cuda.is_available(): _torch.cuda.synchronize()
                        sel_time = _time.perf_counter() - s0
                        if not first_sel_done:
                            first_sel_time = sel_time; first_sel_done = True

                        # select best mask by IoU
                        cand = list(_enumerate_masks_from_anns(anns, H, W, x1,y1,x2,y2))
                        if cand:
                            ious = [mask_iou_binary(mb, gt_bool) for mb in cand]
                            pred_bool = cand[int(_np.argmax(ious))]

                    # ----- Fallback：optional -----
                    if pred_bool is None and masks_tensor is not None and masks_tensor.shape[0] > 0:
                        Ht = 256; Wt = max(1, int(round(256.0 * W / max(1, H))))
                        gt_ds_u8 = safe_resize(gt_bool.astype(_np.uint8), Wt, Ht, _cv2.INTER_NEAREST)
                        if gt_ds_u8 is None:
                            rows.append(("err", ("invalid_resize_gt",), mp, tg)); continue
                        gt_ds = gt_ds_u8.astype(bool)
                        best_iou, best_mask = -1.0, None
                        for k in range(int(masks_tensor.shape[0])):
                            m = masks_tensor[k]
                            if _torch.is_tensor(m):
                                m = m.detach().float().cpu().numpy()
                            m_bin = (m > 0.5).astype(_np.uint8)
                            m_ds_u8 = safe_resize(m_bin, Wt, Ht, _cv2.INTER_NEAREST)
                            if m_ds_u8 is None: continue
                            m_ds = m_ds_u8.astype(bool)
                            iouv = mask_iou_binary(m_ds, gt_ds)
                            if iouv > best_iou:
                                best_iou, best_mask = iouv, m_bin
                        if best_mask is not None:
                            pred_u8 = safe_resize(best_mask, W, H, _cv2.INTER_NEAREST)
                            if pred_u8 is None:
                                rows.append(("err", ("invalid_resize_pred",), mp, tg)); continue
                            pred_bool = pred_u8.astype(bool)

                    if pred_bool is None or _np.count_nonzero(pred_bool) == 0:
                        rows.append(("empty_pred", (0.0, float("inf")), mp, tg)); continue

                    # metrics
                    pred_u8 = pred_bool.astype(_np.uint8)
                    p255 = (pred_u8>0).astype(_np.uint8)*255
                    g255 = (gt_up_u8>0).astype(_np.uint8)*255
                    if p255.sum()==0 or g255.sum()==0:
                        dsc, hd = 0.0, float("inf")
                    else:
                        dsc = float(dice_score(p255, g255)); hd = float(hausdorff(p255, g255))

                    # save mask
                    if save_masks:
                        img_name = os.path.splitext(os.path.basename(image_path))[0]
                        save_path = os.path.join(masks_dir, f"{img_name}--{tg}.npy")
                        try: _np.save(save_path, pred_u8.astype(_np.uint8))
                        except Exception: pass

                    rows.append(("ok", (dsc, hd), mp, tg))

                # time 
                t_first = t_forward + (first_sel_time if first_sel_done else 0.0)
                q_out.put(("batch", {"rows": rows, "t_first": t_first}))
            except Exception as e:
                q_out.put(("batch", {"rows":[("err", (str(e),), "", "")]*len(mask_paths), "t_first": None}))

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
                    return {"rows":[(tag, payload, "", "")], "t_first": None}
            time.sleep(0.01)
        # timeout: terminate & restart
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


# ===================== 6) Summary writer =====================
def _summarize_from_metrics_all(metrics_all_csv, dataset_len, total_masks_all, summary_path):
    """Aggregate metrics_all.csv into summary.txt (Time only; no Prep)."""
    dscs, hds = [], []
    reason_counts = {}
    time_by_image = {}

    if os.path.exists(metrics_all_csv):
        with open(metrics_all_csv, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            idx_time = header.index("Time") if "Time" in header else -1
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:  
                    continue
                # metrics
                try:
                    dscs.append(float(parts[3]))
                except Exception:
                    pass
                hd_str = parts[4].strip().lower()
                if hd_str not in ("inf", "nan", ""):
                    try:
                        hds.append(float(hd_str))
                    except Exception:
                        pass
                # reason
                reason = parts[5].strip()
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                # record first time per image
                if idx_time >= 0 and len(parts) > idx_time:
                    img = parts[1].strip()
                    tstr = parts[idx_time].strip()
                    if img and tstr:
                        try:
                            tval = float(tstr)
                            if img not in time_by_image and np.isfinite(tval):
                                time_by_image[img] = tval
                        except Exception:
                            pass

    def _mean(a): return float(np.mean(a)) if a else None

    avg_dsc = _mean(dscs); avg_hd = _mean(hds)
    times = list(time_by_image.values())
    avg_time = _mean(times)
    thrpt = (len(times) / sum(times)) if times and sum(times) > 0 else None  # img/s

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Summary ===\n")
        f.write(f"Images (total): {dataset_len}\n")
        f.write(f"Masks (total): {total_masks_all}\n")
        f.write(f"[box] Average DSC: {avg_dsc:.4f}\n" if avg_dsc is not None else "[box] Average DSC: N/A\n")
        f.write(f"[box] Average HD: {avg_hd:.2f}\n"  if avg_hd  is not None else "[box] Average HD: N/A\n")
        if avg_time is not None:
            f.write(f"[box] Inference Time (forward+first select): {avg_time:.6f} sec/img\n")
        if thrpt is not None:
            f.write(f"[box] Throughput: {thrpt:.2f} img/s\n")
        f.write("\n=== Counts (all GT/masks) ===\n")
        ok_cnt = reason_counts.pop("ok", 0)
        f.write(f"OK: {ok_cnt}\n")
        for k in sorted(reason_counts.keys()):
            f.write(f"{k}: {reason_counts[k]}\n")


# ===================== 7) Main evaluate() loop =====================
def evaluate_dataset(dataset_name, dataset_root, mapping_file, checkpoint,
                     save_masks=False, num_shards=1, shard_id=0, out_name=None,
                     imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF, iou=DEFAULT_IOU, retina_masks=DEFAULT_RETINA_MASKS):
    if not mapping_file or (not os.path.exists(mapping_file)):
        raise RuntimeError(f"mapping_file must exist: {mapping_file}")
    if not out_name or not str(out_name).strip():
        raise ValueError("out_name must be provided (non-empty).")

    worker = PredictorWorker(
        checkpoint, imgsz=imgsz, conf=conf, iou=iou, retina_masks=retina_masks, timeout=PREDICT_TIMEOUT_SEC
    )

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

    with open(mapping_file, "r", encoding="utf-8") as f:
        image_to_masks = json.load(f)
    image_to_masks = _filter_by_shard(image_to_masks, num_shards, shard_id)

    dataset = GroupedDataset(image_to_masks, dataset_root)

    buf_ok: List[str] = []; buf_all: List[str] = []
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=DATALOADER_WORKERS,
        persistent_workers=(DATALOADER_WORKERS > 0),
        prefetch_factor=(PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None),
        collate_fn=lambda xs: xs[0]
    )

    processed_masks = 0
    for image_path, mask_paths, tags in tqdm(
        loader, desc="Evaluating", dynamic_ncols=True, mininterval=0.5, ascii=True,
        bar_format="{desc}: |{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ):
        base = os.path.splitext(os.path.basename(image_path))[0]
        payload = worker.process_batch(image_path, mask_paths, tags, save_masks, masks_dir)
        rows = payload.get("rows", [])
        t_first = payload.get("t_first", None)
        time_str = (f"{t_first:.6f}" if (t_first is not None) else "")

        # rows are per-mask, already aligned with SAM
        for (status, pl, mask_path_abs, out_tag) in rows:
            if mask_path_abs:
                rel = os.path.relpath(mask_path_abs, os.path.join(dataset_root, "masks")).replace("\\", "/")
                mask_key = os.path.splitext(rel)[0]
            else:
                mask_key = f"{base}--{out_tag or 'NA'}"

            if status == "ok":
                dsc, hd = pl
                buf_ok.append (f"{mask_key},{base},{out_tag},{dsc:.6f},{hd:.6f},{time_str}\n")
                buf_all.append(f"{mask_key},{base},{out_tag},{dsc:.6f},{hd:.6f},ok,{time_str}\n")
            elif status == "empty_pred":
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,empty_pred,{time_str}\n")
            elif status == "empty_gt_read_fail":
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,empty_gt_read_fail,{time_str}\n")
            elif status == "empty_gt_all_zero":
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,empty_gt_all_zero,{time_str}\n")
            elif status == "timeout":
                buf_all.append(f"{mask_key},{base},{out_tag},0.000000,inf,timeout,{time_str}\n")
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
    total_masks_all  = sum(len(v) for v in image_to_masks.values())
    summary_path = os.path.join(output_dir, "summary.txt")
    _summarize_from_metrics_all(metrics_all_csv, total_images_all, total_masks_all, summary_path)

    try:
        with open(os.path.join(output_dir, "done.flag"), "w", encoding="utf-8") as _f:
            _f.write("ok")
    except Exception:
        pass

    worker.close()


# ===================== 8) CLI =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--sample_name", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou",  type=float, default=DEFAULT_IOU)
    parser.add_argument("--retina_masks", action="store_true")
    parser.add_argument("--no_save_npy", action="store_true")
    args = parser.parse_args()

    sample_root = os.path.join(args.base_dir, args.sample_name)

    if not args.mapping_file or (not os.path.exists(args.mapping_file)):
        print(f"Mapping file not found: {args.mapping_file}")
        raise SystemExit(1)

    evaluate_dataset(
        dataset_name=args.sample_name,
        dataset_root=sample_root,
        mapping_file=args.mapping_file,
        checkpoint=args.checkpoint_path,
        save_masks=not args.no_save,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        out_name=args.out_name,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        retina_masks=bool(args.retina_masks),
    )
