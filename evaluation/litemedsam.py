
"""
LiteMedSAM evaluation

Contents:
  1) Setup & globals
  2) preproc
  3) utilities
  4) model
  5) evaluation
  6) summary writer
  7) CLI
  
"""

import os, json, time, hashlib, math, argparse
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.tiny_vit_sam import TinyViT
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

# ---------------- 1) globals & determinism ----------------
def set_deterministic(seed=2024):
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass


# ---------------- 2) preproc (LiteMedSAM) ----------------
def resize_longest_side(img, target=256):
    h, w = img.shape[:2]
    s = float(target) / max(h, w)
    nh, nw = int(h * s + 0.5), int(w * s + 0.5)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def pad_to_square(x, side=256):
    h, w = x.shape[:2]
    ph, pw = side - h, side - w
    if x.ndim == 3:
        return np.pad(x, ((0, ph), (0, pw), (0, 0)))
    return np.pad(x, ((0, ph), (0, pw)))

def minmax01(x):
    mn, mx = float(x.min()), float(x.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x.astype(np.float32) - mn) / (mx - mn + 1e-8)

def tight_bbox(mask_u8):
    """Return tight bbox [x1,y1,x2,y2] on padded-256 grid. (closed interval; no expansion)"""
    if mask_u8 is None: return None
    m = (mask_u8 > 0).astype(np.uint8)
    if m.max() == 0: return None
    ys, xs = np.where(m > 0)
    if ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, x2, y2]


# ---------------- 3) utilities ----------------
def resolve_under(root, sub, relpath):
    rel = str(relpath).replace("\\", "/")
    p0 = os.path.normpath(os.path.join(root, rel))
    if os.path.exists(p0): return p0
    head = sub.lower() + "/"
    rel2 = rel[len(head):] if rel.lower().startswith(head) else rel
    p1 = os.path.normpath(os.path.join(root, sub, rel2))
    if os.path.exists(p1): return p1
    return os.path.normpath(os.path.join(root, sub, os.path.basename(rel2)))

def filter_by_shard(mapping, num_shards, shard_id):
    if num_shards <= 1: return mapping
    out = {}
    for img_rel, masks in mapping.items():
        h = int(hashlib.md5(str(img_rel).encode()).hexdigest(), 16)
        if (h % num_shards) == shard_id:
            out[str(img_rel)] = masks
    return out

def mask_key_from_mapping(mask_rel, dataset_root):
    """
    Convert mapping's mask path to SAM-format key:
    relative to <dataset_root>/masks, without extension.
    """
    abs_path = resolve_under(dataset_root, "masks", mask_rel)
    rel = os.path.relpath(abs_path, os.path.join(dataset_root, "masks")).replace("\\", "/")
    key = os.path.splitext(rel)[0]
    return key


# ---------------- 4) model ----------------
class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder  = image_encoder
        self.mask_decoder   = mask_decoder
        self.prompt_encoder = prompt_encoder

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, orig_size):
        # crop away padding to new_size, then upsample to original image size
        masks = masks[..., :new_size[0], :new_size[1]]
        return F.interpolate(masks, size=(orig_size[0], orig_size[1]),
                             mode="bilinear", align_corners=False)

def build_model(device="cuda"):
    image_encoder = TinyViT(
        img_size=256, in_chans=3,
        embed_dims=[64,128,160,320], depths=[2,2,6,2],
        num_heads=[2,4,5,10], window_sizes=[7,7,14,7],
        mlp_ratio=4., drop_rate=0., drop_path_rate=0.0,
        use_checkpoint=False, mbconv_expand_ratio=4.0,
        local_conv_size=3, layer_lr_decay=0.8
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256, image_embedding_size=(64,64),
        input_image_size=(256,256), mask_in_chans=16
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256,
    )
    model = MedSAM_Lite(image_encoder, mask_decoder, prompt_encoder)
    return model.to(device).eval()

@torch.no_grad()
def predict_one_label(model, img_embed, box_256, new_size, orig_size):
    """
    box_256: np.ndarray of shape (1,4) in padded-256 coord (XYXY, closed)
    returns: np.uint8 mask (H,W) in original image size
    """
    dev = img_embed.device
    box_t = torch.as_tensor(box_256, dtype=torch.float32, device=dev)
    if box_t.ndim == 2:  # (1,4) -> (1,1,4)
        box_t = box_t[:, None, :]

    sp, dn = model.prompt_encoder(points=None, boxes=box_t, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sp,
        dense_prompt_embeddings=dn,
        multimask_output=False
    )
    prob = torch.sigmoid(model.postprocess_masks(logits, new_size, orig_size))
    pred = (prob[0, 0] > 0.5).to(torch.uint8).cpu().numpy()
    return pred


# ---------------- 5) evaluation ----------------
def dice_coef(p, g):
    p = (p > 0).astype(np.uint8)
    g = (g > 0).astype(np.uint8)
    inter = float((p & g).sum())
    den = float(p.sum() + g.sum())
    if den == 0.0:
        return 1.0
    return 2.0 * inter / den

def hausdorff_px(a, b):
    # both {0,1}; if either empty -> inf
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    if a.max() == 0 or b.max() == 0:
        return float("inf")
    da = cv2.distanceTransform((1 - a).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((1 - b).astype(np.uint8), cv2.DIST_L2, 3)
    ha = float(da[b.astype(bool)].max())
    hb = float(db[a.astype(bool)].max())
    return max(ha, hb)

def evaluate_dataset(base_dir, sample_name, mapping_file, checkpoint_path,
                     out_name, shard_id=0, num_shards=1, device="cuda",
                     no_save=False):

    sample_root = os.path.join(base_dir, sample_name)
    out_root = os.path.join(sample_root, out_name)
    out_dir  = os.path.join(out_root, f"shard{shard_id}")
    masks_dir= os.path.join(out_dir, "masks_npy")
    os.makedirs(out_dir, exist_ok=True)
    if not no_save:
        os.makedirs(masks_dir, exist_ok=True)

    # CSVs
    metrics_csv     = os.path.join(out_dir, "metrics.csv")
    metrics_all_csv = os.path.join(out_dir, "metrics_all.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("mask,image,tag,DSC,HD,Time\n")
    if not os.path.exists(metrics_all_csv):
        with open(metrics_all_csv, "w", encoding="utf-8") as f:
            f.write("mask,image,tag,DSC,HD,reason,Time\n")

    # load mapping
    with open(mapping_file, "r", encoding="utf-8") as f:
        img2masks = json.load(f)
    img2masks = filter_by_shard(img2masks, num_shards, shard_id)

    # build model & load weights
    model = build_model(device=device)
    sd = torch.load(checkpoint_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}")

    # iterate images
    total_images = 0
    for img_rel, mask_list in tqdm(
        list(img2masks.items()),
        desc=f"[shard{shard_id}] Evaluating",
        dynamic_ncols=True,
        mininterval=0.5,
        ascii=True,
        bar_format="{desc}: |{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        total_images += 1
        img_abs = resolve_under(sample_root, "images", img_rel)
        base = os.path.splitext(os.path.basename(img_abs))[0]

        # ----- Read + preproc (LiteMedSAM) -----
        bgr = cv2.imread(img_abs, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            # record errors for every GT of this image
            for m_rel in mask_list:
                key = mask_key_from_mapping(m_rel, sample_root)
                tag = os.path.splitext(os.path.basename(resolve_under(sample_root, "masks", m_rel)))[0].split("--")[-1]
                with open(metrics_all_csv, "a", encoding="utf-8") as f:
                    f.write(f"{key},{base},{tag},0.000000,inf,empty_image_read_fail,\n")
            continue

        if bgr.ndim == 2:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        elif bgr.shape[2] == 4:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        H, W = rgb.shape[:2]
        img256 = resize_longest_side(rgb, 256)
        newh, neww = img256.shape[:2]
        img256 = minmax01(img256)
        img256 = pad_to_square(img256, 256)
        img_t = torch.from_numpy(img256).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # ----- set_image timing -----
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            img_embed = model.image_encoder(img_t)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t_set = time.perf_counter() - t0

        # ----- Loop GT masks -----
        t_decode_first = None
        wrote_first_time = False

        for m_rel in mask_list:
            m_abs = resolve_under(sample_root, "masks", m_rel)
            tag = os.path.splitext(os.path.basename(m_abs))[0].split("--")[-1]
            key = mask_key_from_mapping(m_rel, sample_root)

            gt = cv2.imread(m_abs, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                with open(metrics_all_csv, "a", encoding="utf-8") as f:
                    f.write(f"{key},{base},{tag},0.000000,inf,empty_gt_read_fail,\n")
                continue
            if gt.max() == 0:
                with open(metrics_all_csv, "a", encoding="utf-8") as f:
                    f.write(f"{key},{base},{tag},0.000000,inf,empty_gt_all_zero,\n")
                continue

            # GT -> 256 coords (resize to neww,newh, then pad to 256)
            gt_rs = cv2.resize((gt > 0).astype(np.uint8), (neww, newh), interpolation=cv2.INTER_NEAREST)
            gt_pd = pad_to_square(gt_rs, 256).astype(np.uint8)
            box = tight_bbox(gt_pd)
            if box is None:
                with open(metrics_all_csv, "a", encoding="utf-8") as f:
                    f.write(f"{key},{base},{tag},0.000000,inf,empty_gt_all_zero,\n")
                continue

            # predict (time decode for the FIRST mask only)
            if (t_decode_first is None):
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                pred_first = predict_one_label(model, img_embed, np.asarray(box, np.int32)[None, :], (newh, neww), (H, W))
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t_decode_first = time.perf_counter() - t1
                pred = pred_first
            else:
                pred = predict_one_label(model, img_embed, np.asarray(box, np.int32)[None, :], (newh, neww), (H, W))

            # metrics
            gt_bin = (gt > 0).astype(np.uint8)
            if pred.shape != gt_bin.shape:
                pred = cv2.resize(pred, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

            if pred.sum() == 0:
                # empty prediction -> only write to metrics_all.csv
                time_str = f"{(t_set + t_decode_first):.6f}" if (t_decode_first is not None and not wrote_first_time) else ""
                with open(metrics_all_csv, "a", encoding="utf-8") as f:
                    f.write(f"{key},{base},{tag},0.000000,inf,empty_pred,{time_str}\n")
                wrote_first_time = True  # time only once per image
                continue

            dsc = dice_coef(pred, gt_bin)
            hd  = hausdorff_px(pred, gt_bin)
            hd_str = (f"{hd:.6f}" if math.isfinite(hd) else "inf")
            time_str = f"{(t_set + t_decode_first):.6f}" if (t_decode_first is not None and not wrote_first_time) else ""

            # write metrics.csv (OK only) & metrics_all.csv
            with open(metrics_csv, "a", encoding="utf-8") as f:
                f.write(f"{key},{base},{tag},{dsc:.6f},{hd_str},{time_str}\n")
            with open(metrics_all_csv, "a", encoding="utf-8") as f:
                f.write(f"{key},{base},{tag},{dsc:.6f},{hd_str},ok,{time_str}\n")

            # save pred mask (0/1)
            if not no_save:
                try:
                    save_path = os.path.join(masks_dir, f"{base}--{tag}.npy")
                    np.save(save_path, (pred > 0).astype(np.uint8))
                except Exception:
                    pass

            wrote_first_time = True  # ensure Time only recorded once per image

    # summary
    _summarize_from_metrics_all(metrics_all_csv,
                                dataset_len=len(img2masks),
                                total_masks_all=sum(len(v) for v in img2masks.values()),
                                summary_path=os.path.join(out_dir, "summary.txt"))

    # done flag
    try:
        with open(os.path.join(out_dir, "done.flag"), "w", encoding="utf-8") as f:
            f.write("ok")
    except Exception:
        pass


# ---------------- 6) summary writer (aligned with SAM) ----------------
def _summarize_from_metrics_all(metrics_all_csv, dataset_len, total_masks_all, summary_path):
    """
    Aggregate metrics_all.csv into summary.txt
    (Average over all numeric DSC/HD entries, report reason counts and throughput).
    """
    from statistics import mean

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
                # time (first occurrence per image)
                img = parts[1].strip()
                if img and idx_time >= 0 and len(parts) > idx_time:
                    tstr = parts[idx_time].strip()
                    if tstr:
                        try:
                            tval = float(tstr)
                            if img not in time_by_image and np.isfinite(tval):
                                time_by_image[img] = tval
                        except Exception:
                            pass

    avg_dsc = (mean(dscs) if dscs else None)
    avg_hd  = (mean(hds)  if hds  else None)

    times = list(time_by_image.values())
    avg_time = (float(np.mean(times)) if times else None)
    thrpt = (len(times) / sum(times)) if times and sum(times) > 0 else None  # img/s

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Summary ===\n")
        f.write(f"Images (total): {dataset_len}\n")
        f.write(f"Masks (total): {total_masks_all}\n")
        f.write(f"[box] Average DSC: {avg_dsc:.4f}\n" if avg_dsc is not None else "[box] Average DSC: N/A\n")
        f.write(f"[box] Average HD: {avg_hd:.2f}\n"  if avg_hd  is not None else "[box] Average HD: N/A\n")
        if avg_time is not None:
            f.write(f"[box] Inference Time (set_image+first decode): {avg_time:.6f} sec/img\n")
        if thrpt is not None:
            f.write(f"[box] Throughput: {thrpt:.2f} img/s\n")

        f.write("\n=== Counts (all GT/masks) ===\n")
        ok_cnt = reason_counts.pop("ok", 0)
        f.write(f"OK: {ok_cnt}\n")
        for k in sorted(reason_counts.keys()):
            f.write(f"{k}: {reason_counts[k]}\n")


# ---------------- 7) CLI ----------------
def main():
    ap = argparse.ArgumentParser("LiteMedSAM eval (SAM-format outputs; tight bbox; official preprocess)")
    # required by run_and_merge_generic.py
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--model_type", type=str, default="vit_t")  # accepted for runner; not used
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--sample_name", type=str, required=True)
    ap.add_argument("--mapping_file", type=str, required=True)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--out_name", type=str, required=True)
    ap.add_argument("--no_save", action="store_true")
    # extras
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    set_deterministic(2024)
    evaluate_dataset(
        base_dir=args.base_dir,
        sample_name=args.sample_name,
        mapping_file=args.mapping_file,
        checkpoint_path=args.checkpoint_path,
        out_name=args.out_name,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        device=args.device,
        no_save=args.no_save,
    )

if __name__ == "__main__":
    main()
