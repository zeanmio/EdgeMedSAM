
"""
SAM-Med2D evaluation

Contents:
  1) Imports & globals
  2) Utilities 
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
import time
import torch
import numpy as np
import hashlib
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from multiprocessing import get_context

from utils.metrics import dice_score, hausdorff

# ===== 1) Parameters =====
PREDICT_TIMEOUT_SEC      = 60
DATALOADER_WORKERS       = 0
PREFETCH_FACTOR          = 2
RESTART_WORKER_EVERY_N   = 2000
FLUSH_EVERY              = 500
MODEL_LOAD_MAX_WAIT_SEC  = 180
# =============================================

# Default image size
DEFAULT_IMAGE_SIZE = 1024

# ---- Reproducibility / numeric stability ----
try:
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
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


class GroupedMed2DDataset(Dataset):
    """Yield (image_abs, [mask_abs], [tags]) per image."""
    def __init__(self, image_to_masks, dataset_root):
        self.dataset_root = dataset_root
        self.items = [(img_path, mask_list) for img_path, mask_list in image_to_masks.items()]
    def __len__(self): return len(self.items)
    def _resolve_image_path(self, img_rel): return resolve_under(self.dataset_root, "images", str(img_rel))
    def _resolve_mask_path(self, mask_rel): return resolve_under(self.dataset_root, "masks", str(mask_rel))
    def __getitem__(self, idx):
        image_rel, mask_rel_paths = self.items[idx]
        image_full_path = self._resolve_image_path(image_rel)
        masks, tags = [], []
        for mask_rel_path in mask_rel_paths:
            mask_path = self._resolve_mask_path(mask_rel_path)
            tag = os.path.splitext(os.path.basename(mask_path))[0].split("--")[-1]
            masks.append(mask_path); tags.append(tag)
        return image_full_path, masks, tags


def _filter_by_shard(mapping, num_shards, shard_id):
    """Deterministic image-level sharding by md5 hash."""
    if num_shards <= 1:
        return mapping
    out = {}
    for img_rel, masks in mapping.items():
        h = int(hashlib.md5(str(img_rel).encode()).hexdigest(), 16)
        if (h % num_shards) == shard_id:
            out[str(img_rel)] = masks
    return out

# ==================== 3) Predictor Worker ====================

class PredictorWorker:
    def __init__(self, checkpoint, model_type, sam_init_checkpoint="", timeout=PREDICT_TIMEOUT_SEC):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.sam_init_checkpoint = sam_init_checkpoint
        self.timeout = timeout
        self.q_in = None; self.q_out = None; self.p = None
        self.num_params = None
        self._start()

    def _start(self):
        self.q_in = CTX.Queue(1); self.q_out = CTX.Queue(1)
        self.p = CTX.Process(
            target=self._worker,
            args=(self.q_in, self.q_out, self.checkpoint, self.model_type, self.sam_init_checkpoint)
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
    def _worker(q_in, q_out, checkpoint, model_type, sam_init_checkpoint):
        import os, sys, cv2, numpy as np, torch, time as _time, importlib.util, math
        import torch.nn.functional as F

        PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        PIXEL_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        def preprocess_bgr_to_tensor(bgr, image_size, device):
            H, W = bgr.shape[:2]
            img = (bgr.astype(np.float32) - PIXEL_MEAN) / PIXEL_STD
            if H < image_size and W < image_size:
                top = (image_size - H) // 2
                bottom = image_size - H - top
                left = (image_size - W) // 2
                right = image_size - W - left
                img256 = cv2.copyMakeBorder(img, top, bottom, left, right,
                                            borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
                pad = (top, left)
            else:
                img256 = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                pad = None
            img_t = torch.from_numpy(img256).permute(2,0,1).unsqueeze(0).to(device=device, dtype=torch.float32)
            return img_t, pad, (H, W)

        def map_boxes_img_to_input_torch(boxes_xyxy_img, ori_size, image_size, pad, device, dtype=torch.float32):
            H, W = ori_size
            boxes = boxes_xyxy_img.to(device=device, dtype=dtype)
            if H < image_size and W < image_size and (pad is not None):
                top, left = pad
                off = torch.tensor([left, top, left, top], device=device, dtype=dtype)
                boxes_in = boxes + off
            else:
                sx = image_size / float(W)
                sy = image_size / float(H)
                scale = torch.tensor([sx, sy, sx, sy], device=device, dtype=dtype)
                boxes_in = boxes * scale
            boxes_in[:, 0::2] = boxes_in[:, 0::2].clamp_(0, image_size - 1)  # x1,x2
            boxes_in[:, 1::2] = boxes_in[:, 1::2].clamp_(0, image_size - 1)  # y1,y2
            return boxes_in

        @torch.no_grad()
        def postprocess_lowres_to_orig(low_res_masks, original_size, image_size, pad, thr=0.5):
            H, W = original_size
            m = F.interpolate(low_res_masks, (image_size, image_size), mode="bilinear", align_corners=False)
            if H < image_size and W < image_size and (pad is not None):
                top, left = pad
                m = m[..., top:top+H, left:left+W]
            else:
                m = F.interpolate(m, (H, W), mode="bilinear", align_corners=False)
            m = (torch.sigmoid(m) > thr).to(torch.uint8)  # (N,1,H,W)
            return m[:, 0].cpu().numpy()

        # =====================================================

        def _import_sammed2d_sa_module(sa_dir: str):
            init_py = os.path.join(sa_dir, "__init__.py")
            if not os.path.exists(init_py):
                raise RuntimeError(f"Not found: {init_py}")
            spec = importlib.util.spec_from_file_location("segment_anything", init_py)
            module = importlib.util.module_from_spec(spec)
            sys.modules.pop("segment_anything", None)
            sys.modules["segment_anything"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module

        sa_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Model", "SAM-Med2D", "segment_anything"))
        try:
            sa = _import_sammed2d_sa_module(sa_dir)
        except Exception as e:
            q_out.put(("ready", None))
            while True:
                job = q_in.get()
                if job is None: break
                q_out.put(("batch", {"results":[("err", (f"import_error:{e}",))]*len(job[1]),
                                     "t_first": None, "t_all": 0.0, "prompts": 0, "t_prep": 0.0}))
            return

        sam_model_registry = sa.sam_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try to infer model image_size from checkpoint(s)
        def _peek_state_dict_shape(ckpt_path):
            try:
                sd = torch.load(ckpt_path, map_location="cpu")
                if isinstance(sd, dict):
                    for k in ("model", "state_dict", "ema"):
                        if k in sd and isinstance(sd[k], dict):
                            sd = sd[k]
                if not isinstance(sd, dict):
                    return None, None
                pe = sd.get("image_encoder.pos_embed", None)
                pw = sd.get("image_encoder.patch_embed.proj.weight", None)
                if pe is None or pw is None:
                    return None, None
                if pe.ndim == 4:      # [1,H,W,C]
                    H = int(pe.shape[1])
                elif pe.ndim == 3:    # [1,HW,C]
                    H = int(round((pe.shape[1]) ** 0.5))
                else:
                    return None, None
                patch = int(pw.shape[-1])
                return H, patch
            except Exception:
                return None, None

        h_med, p_med = _peek_state_dict_shape(checkpoint) if checkpoint else (None, None)
        h_sam, p_sam = _peek_state_dict_shape(sam_init_checkpoint) if sam_init_checkpoint else (None, None)
        if h_med and p_med:
            target_imgsz = max(16, h_med) * p_med
            src_from = "med2d"
        elif h_sam and p_sam:
            target_imgsz = max(16, h_sam) * p_sam
            src_from = "sam"
        else:
            target_imgsz = DEFAULT_IMAGE_SIZE
            src_from = "default"
        sys.stderr.write(f"[sammed2d] ctor target image_size={target_imgsz} (from {src_from})\n")

        # Build model
        from types import SimpleNamespace
        ctor = sam_model_registry[model_type]
        model = None
        ctor_error = None
        try:
            if sam_init_checkpoint:
                args_obj = SimpleNamespace(
                    image_size=target_imgsz,
                    vit=model_type,
                    sam_checkpoint=sam_init_checkpoint,
                    encoder_adapter=True,
                    freeze_image_encoder=False,
                    freeze_prompt_encoder=False,
                    freeze_mask_decoder=False,
                )
                model = ctor(args_obj)
            else:
                try:
                    model = ctor(SimpleNamespace(image_size=target_imgsz))
                except Exception as e:
                    ctor_error = e
        except Exception as e:
            ctor_error = e

        if model is None:
            q_out.put(("ready", None))
            while True:
                job = q_in.get()
                if job is None: break
                msg = f"ctor_failed:{type(ctor_error).__name__}:{ctor_error}; need_official_sam_ckpt: pass --sam_init_checkpoint"
                q_out.put(("batch", {"results":[("err", (msg,))]*len(job[1]),
                                     "t_first": None, "t_all": 0.0, "prompts": 0, "t_prep": 0.0}))
            return

        # Robust (shape-tolerant) state_dict loading with basic interpolation
        def _force_load(m, ckpt_path, dev):
            sd = torch.load(ckpt_path, map_location=dev)
            if isinstance(sd, dict):
                for k in ("model", "state_dict", "ema"):
                    if k in sd and isinstance(sd[k], dict):
                        sd = sd[k]
            if isinstance(sd, dict) and sd and isinstance(next(iter(sd)), str):
                ks = list(sd.keys())
                if ks[0].startswith("module."):
                    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                if any(k.startswith("sam.") for k in ks):
                    sd = {k.replace("sam.", "", 1): v for k, v in sd.items()}
                if any(k.startswith("model.") for k in ks):
                    sd = {k.replace("model.", "", 1): v for k, v in sd.items()}

            msd = m.state_dict()
            to_drop = []

            for k, v in list(sd.items()):
                if k not in msd:
                    continue
                if msd[k].shape == v.shape:
                    continue
                try:
                    if "image_encoder.pos_embed" in k or "pos_embed" in k:
                        if v.ndim == 4 and msd[k].ndim == 4:
                            v2 = F.interpolate(
                                v.permute(0, 3, 1, 2).float(),
                                size=msd[k].shape[1:3],
                                mode="bilinear",
                                align_corners=False
                            ).permute(0, 2, 3, 1).to(msd[k].dtype)
                            sd[k] = v2; continue
                        elif v.ndim == 3:
                            s = int(round(math.sqrt(v.shape[1])))
                            if msd[k].ndim == 3:
                                t = int(round(math.sqrt(msd[k].shape[1])))
                                v2 = v.reshape(1, s, s, v.shape[-1]).permute(0, 3, 1, 2).float()
                                v2 = F.interpolate(v2, size=(t, t), mode="bilinear", align_corners=False)
                                v2 = v2.permute(0, 2, 3, 1).reshape(1, t*t, v.shape[-1]).to(msd[k].dtype)
                                sd[k] = v2; continue
                            else:
                                tH, tW = msd[k].shape[1], msd[k].shape[2]
                                v2 = v.reshape(1, s, s, v.shape[-1]).permute(0, 3, 1, 2).float()
                                v2 = F.interpolate(v2, size=(tH, tW), mode="bilinear", align_corners=False)
                                v2 = v2.permute(0, 2, 3, 1).to(msd[k].dtype)
                                sd[k] = v2; continue

                    if (("rel_pos_h" in k) or ("rel_pos_w" in k)) and v.ndim == 2 and msd[k].ndim == 2:
                        v2 = F.interpolate(
                            v.unsqueeze(0).permute(0, 2, 1).float(),
                            size=(msd[k].shape[0],),
                            mode="linear",
                            align_corners=False
                        ).permute(0, 2, 1).squeeze(0).to(msd[k].dtype)
                        sd[k] = v2; continue
                except Exception:
                    pass
                to_drop.append(k)

            for k in to_drop:
                sd.pop(k, None)

            try:
                missing, unexpected = m.load_state_dict(sd, strict=False)
            except Exception:
                sd2 = {k: v for k, v in sd.items() if (k in msd and msd[k].shape == v.shape)}
                missing, unexpected = m.load_state_dict(sd2, strict=False)

            matched = sum((k in msd) and (msd[k].shape == sd.get(k, torch.empty(0)).shape) for k in sd.keys())
            total   = len(msd)
            sys.stderr.write(
                f"[sammed2d] load ckpt matched={matched}/{total}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}, dropped={len(to_drop)}\n"
            )
            return matched, missing, unexpected

        _force_load(model, checkpoint, device)

        model = model.to(device).eval()
        image_size = getattr(getattr(model, "image_encoder", None), "img_size", None) or target_imgsz or 256

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        q_out.put(("ready", num_params))

        def _compute_bbox_from_mask(mask_u8):
            if mask_u8 is None or mask_u8.max() == 0:
                return None
            ys, xs = np.where(mask_u8 > 0)
            if ys.size == 0:
                return None
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            return [x1, y1, x2, y2]

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

                img_t, pad, (H_img, W_img) = preprocess_bgr_to_tensor(bgr, image_size, device)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_prep_total = _time.perf_counter() - t0

                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = _time.perf_counter()
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
                    img_embed = model.image_encoder(img_t)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_set = _time.perf_counter() - t0

                results = [None] * len(mask_paths)
                valid_flags = []
                boxes_img, gts = [], []
                for idx, mask_path in enumerate(mask_paths):
                    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if gt is None:
                        results[idx] = ("empty_gt_read_fail", (0.0, float("inf")))
                        valid_flags.append(False); gts.append(None); boxes_img.append(None); continue
                    if gt.max() == 0:
                        results[idx] = ("empty_gt_all_zero", (0.0, float("inf")))
                        valid_flags.append(False); gts.append(gt); boxes_img.append(None); continue

                    bbox_m = _compute_bbox_from_mask(((gt > 0).astype(np.uint8) * 255))
                    if bbox_m is None:
                        results[idx] = ("empty_gt_all_zero", (0.0, float("inf")))
                        valid_flags.append(False); gts.append(gt); boxes_img.append(None); continue

                    Hgt, Wgt = gt.shape[:2]
                    sx = W_img / float(Wgt); sy = H_img / float(Hgt)
                    x1, y1, x2, y2 = bbox_m
                    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
                    X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
                    X1 = max(0, min(X1, W_img - 1)); X2 = max(0, min(X2, W_img - 1))
                    Y1 = max(0, min(Y1, H_img - 1)); Y2 = max(0, min(Y2, H_img - 1))
                    if X2 < X1: X1, X2 = X2, X1
                    if Y2 < Y1: Y1, Y2 = Y2, Y1
                    boxes_img.append([X1, Y1, X2, Y2]); gts.append(gt); valid_flags.append(True)

                true_indices = [i for i, v in enumerate(valid_flags) if v]
                t_first = None; t_all = 0.0; prompts = 0

                if true_indices:
                    boxes_img_t = torch.tensor([boxes_img[i] for i in true_indices], dtype=torch.float32, device=device)
                    boxes_in_t = map_boxes_img_to_input_torch(boxes_img_t, (H_img, W_img), image_size, pad, device)

                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    d0 = _time.perf_counter()
                    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
                        sp1, dn1 = model.prompt_encoder(points=None, boxes=boxes_in_t[0:1], masks=None)
                        low1, iou1 = model.mask_decoder(
                            image_embeddings=img_embed,
                            image_pe=model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sp1,
                            dense_prompt_embeddings=dn1,
                            multimask_output=False,
                        )
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    dt_first = _time.perf_counter() - d0
                    t_first = t_set + dt_first
                    t_all += dt_first; prompts += 1

                    if boxes_in_t.shape[0] > 1:
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        d0 = _time.perf_counter()
                        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
                            sp, dn = model.prompt_encoder(points=None, boxes=boxes_in_t[1:], masks=None)
                            low_rest, iou_rest = model.mask_decoder(
                                image_embeddings=img_embed,
                                image_pe=model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sp,
                                dense_prompt_embeddings=dn,
                                multimask_output=False,
                            )
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        t_all += (_time.perf_counter() - d0); prompts += (boxes_in_t.shape[0] - 1)
                        low_all = torch.cat([low1, low_rest], dim=0)  # (N,1,h,w)
                    else:
                        low_all = low1  # (1,1,h,w)

                    preds_np = postprocess_lowres_to_orig(low_all, (H_img, W_img), image_size, pad, thr=0.5)

                    for out_idx, mi in enumerate(true_indices):
                        pred = preds_np[out_idx]; gt = gts[mi]
                        if pred.sum() == 0:
                            results[mi] = ("empty_pred", (0.0, float("inf"))); continue
                        if pred.shape != gt.shape:
                            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                        p255 = (pred > 0).astype(np.uint8) * 255
                        g255 = (gt   > 0).astype(np.uint8) * 255
                        if p255.sum() == 0 or g255.sum() == 0:
                            dsc, hd = 0.0, float("inf")
                        else:
                            dsc = float(dice_score(p255, g255)); hd = float(hausdorff(p255, g255))
                        results[mi] = ("ok", (dsc, hd))

                for i in range(len(results)):
                    if results[i] is None:
                        results[i] = ("err", ("unknown_state",))

                q_out.put(("batch", {"results": results, "t_first": t_first, "t_all": t_all,
                                     "prompts": prompts, "t_prep": t_prep_total}))
            except Exception as e:
                q_out.put(("batch", {"results":[("err", (str(e),))]*len(mask_paths),
                                     "t_first": None, "t_all": 0.0, "prompts": 0, "t_prep": 0.0}))

    def process_batch(self, image_path, mask_paths, tags, save_masks, masks_dir):
        """Send job + wait with timeout; auto-restart on timeout/failure."""
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
        """Periodic soft restart (e.g., after N masks)."""
        try:
            if self.p and self.p.is_alive():
                self.q_in.put(None); self.p.join(timeout=1)
        except Exception:
            pass
        self._start()

    def close(self):
        """Shutdown child process."""
        try:
            if self.p and self.p.is_alive():
                self.q_in.put(None); self.p.join(timeout=1)
        except Exception:
            pass

# ==================== 4) Summary writer ====================

def _summarize_from_metrics_all(metrics_all_csv, dataset_len, total_masks_all, summary_path):
    dscs_all, hds_all = [], []
    reason_counts = {}
    time_by_image, prep_by_image = {}, {}

    dscs_ok, hds_ok = [], []
    from collections import defaultdict
    img2dscs, img2hds = defaultdict(list), defaultdict(list)
    imgtag2dscs, imgtag2hds = defaultdict(list), defaultdict(list)

    if os.path.exists(metrics_all_csv):
        import csv as _csv
        with open(metrics_all_csv, "r", encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            fns = [ (fn or "").strip() for fn in (reader.fieldnames or []) ]
            fmap = {fn.lower(): fn for fn in fns}

            def _pick(row, key, default=""):
                k = fmap.get(key.lower())
                return (row.get(k, default) if k else default)

            def _pfloat(s):
                try: return float(s)
                except Exception: return None

            for row in reader:
                img = (_pick(row, "image").strip())
                tag = (_pick(row, "tag").strip())
                rsn = (_pick(row, "reason").strip())

                d_all = _pfloat(_pick(row, "DSC") or _pick(row, "dsc"))
                if d_all is not None:
                    dscs_all.append(d_all)

                hd_raw = (_pick(row, "HD") or _pick(row, "hd")).strip().lower()
                if hd_raw not in ("", "inf", "nan"):
                    try:
                        hds_all.append(float(h_raw := hd_raw))
                    except Exception:
                        pass

                reason_counts[rsn] = reason_counts.get(rsn, 0) + 1

                if img:
                    tstr = (_pick(row, "Time") or _pick(row, "time") or "").strip()
                    if tstr:
                        tval = _pfloat(tstr)
                        if (tval is not None) and np.isfinite(tval) and (img not in time_by_image):
                            time_by_image[img] = tval
                    pstr = (_pick(row, "Prep") or _pick(row, "prep") or "").strip()
                    if pstr:
                        pval = _pfloat(pstr)
                        if (pval is not None) and np.isfinite(pval) and (img not in prep_by_image):
                            prep_by_image[img] = pval

                if rsn == "ok":
                    if d_all is not None:
                        dscs_ok.append(d_all)
                        if img: img2dscs[img].append(d_all)
                        if img and tag: imgtag2dscs[(img, tag)].append(d_all)
                    if hd_raw not in ("", "inf", "nan"):
                        try:
                            h = float(hd_raw)
                            if img: img2hds[img].append(h)
                            if img and tag: imgtag2hds[(img, tag)].append(h)
                        except Exception:
                            pass

    def _safe_mean(arr): return (float(np.mean(arr)) if arr else None)
    def _safe_median(arr):
        return (float(np.median(np.asarray(arr, dtype=float))) if arr else None)
    def _pct(arr, p):
        if not arr: return None
        a = np.asarray(arr, dtype=float)
        if a.size == 0: return None
        try: return float(np.percentile(a, p, method="linear"))
        except TypeError: return float(np.percentile(a, p, interpolation="linear"))

    avg_dsc_all = _safe_mean(dscs_all)
    avg_hd_all  = _safe_mean(hds_all)

    times = list(time_by_image.values())
    preps = list(prep_by_image.values())
    avg_time = _safe_mean(times)
    avg_prep = _safe_mean(preps)
    p_time = {q: _pct(times, q) for q in (50, 90, 95, 99)} if times else {}
    p_prep = {q: _pct(preps, q) for q in (50, 90, 95, 99)} if preps else {}
    thrpt_model = (len(times) / sum(times)) if times and (sum(times) > 0) else None

    img_meds_dsc = [_safe_median(v) for v in img2dscs.values() if v]
    img_meds_hd  = [_safe_median(v) for v in img2hds.values()  if v]
    img_med_of_meds_dsc = _safe_median([x for x in img_meds_dsc if x is not None])
    img_med_of_meds_hd  = _safe_median([x for x in img_meds_hd  if x is not None])

    from collections import defaultdict as _dd
    tag2imgmeds_dsc, tag2imgmeds_hd = _dd(list), _dd(list)
    for (img, tag), vals in imgtag2dscs.items():
        m = _safe_median(vals)
        if m is not None: tag2imgmeds_dsc[tag].append(m)
    for (img, tag), vals in imgtag2hds.items():
        m = _safe_median(vals)
        if m is not None: tag2imgmeds_hd[tag].append(m)
    tag_meds_dsc = [_safe_median(v) for v in tag2imgmeds_dsc.values() if v]
    tag_meds_hd  = [_safe_median(v) for v in tag2imgmeds_hd.values()  if v]
    tag_med_of_meds_dsc = _safe_median([x for x in tag_meds_dsc if x is not None])
    tag_med_of_meds_hd  = _safe_median([x for x in tag_meds_hd  if x is not None])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Summary ===\n")
        f.write(f"Images (total): {dataset_len}\n")
        f.write(f"Masks (total): {total_masks_all}\n")
        f.write(f"[box] Average DSC (all rows): {avg_dsc_all:.4f}\n" if avg_dsc_all is not None else "[box] Average DSC (all rows): N/A\n")
        f.write(f"[box] Average HD  (all finite): {avg_hd_all:.2f}\n" if avg_hd_all is not None else "[box] Average HD (all finite): N/A\n")

        if avg_time is not None:
            f.write(f"[box] Inference Time (Model-only): {avg_time:.6f} sec/img\n")
            if thrpt_model is not None:
                f.write(f"[box] Throughput (Model-only): {thrpt_model:.2f} img/s\n")
            if p_time:
                f.write("[box] Inference Latency p50/p90/p95/p99: " + "/".join(f"{p_time[q]:.6f}" for q in (50,90,95,99)) + " sec\n")

        if avg_prep is not None:
            f.write(f"[box] Prep Time: {avg_prep:.6f} sec/img\n")
            if p_prep:
                f.write("[box] Prep Latency p50/p90/p95/p99: " + "/".join(f"{p_prep[q]:.6f}" for q in (50,90,95,99)) + " sec\n")

        f.write("\n=== OK-only Averages ===\n")
        ok_mean_dsc = _safe_mean(dscs_ok)
        ok_mean_hd  = _safe_mean(hds_ok)
        f.write(f"DSC (mean over OK rows): {ok_mean_dsc:.4f}\n" if ok_mean_dsc is not None else "DSC (mean over OK rows): N/A\n")
        f.write(f"HD  (mean over OK finite): {ok_mean_hd:.2f}\n" if ok_mean_hd is not None else "HD  (mean over OK finite): N/A\n")

        f.write("\n=== Medians (OK-only) — Task-style ===\n")
        f.write(
            f"Per-Image median DSC  → overall median: {img_med_of_meds_dsc:.4f} (images used: {len(img_meds_dsc)})\n"
            if img_med_of_meds_dsc is not None else "Per-Image median DSC  → overall median: N/A\n"
        )
        f.write(
            f"Per-Image median HD   → overall median: {img_med_of_meds_hd:.2f} (images used: {len(img_meds_hd)})\n"
            if img_med_of_meds_hd is not None else "Per-Image median HD   → overall median: N/A\n"
        )
        f.write(
            f"Per-Tag   median DSC  → overall median: {tag_med_of_meds_dsc:.4f} (tags used: {len(tag_meds_dsc)})\n"
            if tag_med_of_meds_dsc is not None else "Per-Tag   median DSC  → overall median: N/A\n"
        )
        f.write(
            f"Per-Tag   median HD   → overall median: {tag_med_of_meds_hd:.2f} (tags used: {len(tag_meds_hd)})\n"
            if tag_med_of_meds_hd is not None else "Per-Tag   median HD   → overall median: N/A\n"
        )

        f.write("\n=== Counts (all GT) ===\n")
        ok_cnt = reason_counts.get("ok", 0)
        f.write(f"ok: {ok_cnt}\n")
        for k in sorted(k for k in reason_counts.keys() if k != "ok"):
            f.write(f"{k}: {reason_counts[k]}\n")

# ==================== 5) Main ====================

def evaluate_dataset(dataset_name, dataset_root, mapping_file, checkpoint,
                     model_type="vit_b", save_masks=False,
                     num_shards=1, shard_id=0, sam_init_checkpoint="", out_name="predictions-sammed2d"):
    if not mapping_file or (not os.path.exists(mapping_file)):
        raise RuntimeError(f"mapping_file: {mapping_file}")

    worker = PredictorWorker(checkpoint, model_type,
                             sam_init_checkpoint=sam_init_checkpoint,
                             timeout=PREDICT_TIMEOUT_SEC)

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
            f.write("mask,image,tag,DSC,HD,Time,Prep\n")
    if not os.path.exists(metrics_all_csv):
        with open(metrics_all_csv, "w", encoding="utf-8") as f:
            f.write("mask,image,tag,DSC,HD,reason,Time,Prep\n")

    with open(mapping_file, 'r', encoding="utf-8") as f:
        image_to_masks = json.load(f)
    image_to_masks = _filter_by_shard(image_to_masks, num_shards, shard_id)
    dataset = GroupedMed2DDataset(image_to_masks, dataset_root)

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

        # Copy per-image Time/Prep to all GT rows of that image
        for (status, pl), mask_path, tag in zip(results, mask_paths, tags):
            mask_key = os.path.splitext(os.path.basename(mask_path))[0]
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

# ==================== 6) CLI ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--sam_init_checkpoint", type=str, default="",
                        help="Official SAM ckpt for ctor warm-start (i.e. sam_vit_b_01ec64.pth).")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--sample_name", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--out_name", type=str, default="predictions-sammed2d")
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
        model_type=args.model_type,
        save_masks=not args.no_save,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        sam_init_checkpoint=args.sam_init_checkpoint,
        out_name=args.out_name,
    )
