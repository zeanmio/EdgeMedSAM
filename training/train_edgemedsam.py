"""
Finetune EdgeMedSAM with LoRA on mixed 50k training set (split 811) in 3 stages:

# Stage 1 — Adapter warm‑up
python .\training\train_edgemedsam.py `
  --base_dir  ".\data\mix_50k_seed1" `
  --train_json "split_811\train.json" `
  --val_json   "split_811\val.json" `
  --checkpoint ".\output_lora\edgesam_50k_passA.pth" `
  --model_type edge_sam `
  --out_path   ".\output_lora\edgesam_50k_mp_s1_best.pth" `
  --last_n_blocks 1 --r 8 --alpha 8 --lora_dropout 0.05 `
  --epochs 3 --epoch_samples 4000 `
  --lr 2e-5 --weight_decay 5e-4 --use_l2sp 1e-4 `
  --bbox_from_gt --resize_longest_side 1024 --amp `
  --freeze_bn `
  --workers 4 --prefetch_factor 4 --persistent_workers `
  --save_every_epoch `
  --val_every_steps 2000 `
  --select_by dice --early_stop_patience 2 --early_stop_min_delta 1e-3 `
  --bce_pos_weight 1.0 `
  --balance_by_modality


# Stage 2 — Consolidation
python .\training\train_edgemedsam.py `
  --base_dir  ".\data\mix_50k_seed1" `
  --train_json "split_811\train.json" `
  --val_json   "split_811\val.json" `
  --checkpoint ".\output_lora\edgesam_50k_mp_s1_best.pth" `
  --model_type edge_sam `
  --out_path   ".\output_lora\edgesam_50k_mp_s2_best.pth" `
  --last_n_blocks 0 --r 8 --alpha 8 --lora_dropout 0.10 `
  --epochs 3 --epoch_samples 4000 `
  --lr 2e-5 --weight_decay 5e-4 --use_l2sp 1e-4 `
  --bbox_from_gt --resize_longest_side 1024 --amp `
  --freeze_bn `
  --workers 4 --prefetch_factor 4 --persistent_workers `
  --save_every_epoch `
  --val_every_steps 2000 `
  --select_by dice --early_stop_patience 2 --early_stop_min_delta 1e-3 `
  --bce_pos_weight 2.0 `
  --balance_by_modality

# e02 is the best one then

# Stage 3 — Targeted repair
python .\training\train_edgemedsam.py `
  --base_dir  ".\data\mix_50k_seed1" `
  --train_json "split_811\train.json" `
  --val_json   "split_811\val.json" `
  --checkpoint ".\output_lora\edgesam_50k_mp_s2_best_e02.pth" `
  --model_type edge_sam `
  --out_path   ".\output_lora\edgemedsam_final.pth" `
  --last_n_blocks 1 --r 8 --alpha 8 --lora_dropout 0.05 `
  --epochs 3 --epoch_samples 6000 `
  --lr 1e-5 --weight_decay 5e-4 --use_l2sp 1e-4 `
  --bbox_from_gt --resize_longest_side 1024 --amp `
  --freeze_bn --bn_calibrate_iters 400 `
  --workers 4 --prefetch_factor 4 --persistent_workers `
  --save_every_epoch `
  --val_every_steps 2000 `
  --select_by dice --early_stop_patience 2 --early_stop_min_delta 1e-3 `
  --bce_pos_weight 2.0 `
  --balance_by_modality


# Evaluation on test set

# ===== Dataset (test set) =====
$BASE   = ".\data\mix_50k_seed1"
$SAMPLE = "."
$MAP    = ".\data\mix_50k_seed1\split_811\test_map_for_eval.json"

# ===== EdgeMedSAM (ours) on test =====
python .\evaluation\run_and_merge_generic.py `
  --eval_script ".\evaluation\edgesam.py" `
  --out_name "predictions-edgemedsam_final" `
  --checkpoint_path ".\output_lora\edgemedsam_final.pth" `
  --model_type "edge_sam" `
  --base_dir $BASE `
  --sample_name $SAMPLE `
  --mapping_file $MAP `
  --num_shards 8 `
  --gpus 0 `
  --max_parallel 1 `
  --no_save_npy `
  --skip_done

# ===== SAM Vit-B on test =====
python .\evaluation\run_and_merge_generic.py `
  --eval_script ".\evaluation\sam.py" `
  --out_name "predictions-sam-vit-b-test" `
  --checkpoint_path ".\Model\segment-anything\checkpoints\sam_vit_b.pth" `
  --model_type vit_b `
  --base_dir $BASE `
  --sample_name $SAMPLE `
  --mapping_file $MAP `
  --num_shards 8 `
  --gpus "0" `
  --max_parallel 1 `
  --no_save_npy `
  --skip_done

# ===== EfficientSAM Vit-T on test =====
python .\evaluation\run_and_merge_generic.py `
  --eval_script ".\evaluation\efficientsam.py" `
  --out_name "predictions-efficientsam-vit-t-test" `
  --checkpoint_path ".\Model\EfficientSAM\efficient_sam_vitt.pt" `
  --model_type "vit_t" `
  --resize_long 1024 `
  --base_dir $BASE `
  --sample_name $SAMPLE `
  --mapping_file $MAP `
  --num_shards 8 `
  --gpus 0 `
  --max_parallel 1 `
  --no_save_npy `
  --skip_done
  
# ===== EdgeSAM on test =====
python .\evaluation\run_and_merge_generic.py `
  --eval_script ".\evaluation\edgesam.py" `
  --out_name "predictions-edge-test" `
  --checkpoint_path ".\Model\EdgeSAM\edge_sam.pth" `
  --model_type "edge_sam" `
  --base_dir $BASE `
  --sample_name $SAMPLE `
  --mapping_file $MAP `
  --num_shards 8 `
  --gpus 0 `
  --max_parallel 1 `
  --no_save_npy `
  --skip_done

# ===== SAM-for-Med (vit-b) on test =====
python .\evaluation\run_and_merge_generic.py `
  --eval_script ".\evaluation\sam_for_med.py" `
  --out_name "predictions-sam-for-med-vit-b-test" `
  --checkpoint_path ".\Model\Segment-Anything-Model-for-Medical-Images\medsam_box_best_vitb.pth" `
  --model_type "vit_b" `
  --base_dir $BASE `
  --sample_name $SAMPLE `
  --mapping_file $MAP `
  --num_shards 8 `
  --gpus 0 `
  --max_parallel 1 `
  --no_save_npy `
  --skip_done
  
"""



import os, sys, json, math, random, argparse, time, copy
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

# ========= Add Model/EdgeSAM to sys.path (same as your evaluator) =========
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EDGE_SAM_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "Model", "EdgeSAM"))
if _EDGE_SAM_ROOT not in sys.path:
    sys.path.insert(0, _EDGE_SAM_ROOT)
# ==========================================================================

# ------------------------------
# Utils: seed & device
# ------------------------------
def set_seed(seed: int = 1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_worker(worker_id: int):
    # make dataloader workers deterministic
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

# ------------------------------
# ResizeLongestSide + pad-to-square 
# ------------------------------
class ResizeLongestSide:
    def __init__(self, target_length: int = 1024):
        self.target_length = target_length

    def apply_image(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int]]:
        h, w = img.shape[:2]
        scale = self.target_length / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = np.array(Image.fromarray(img).resize((nw, nh), resample=Image.BILINEAR))
        # zero-pad to square
        pad_h, pad_w = self.target_length - nh, self.target_length - nw
        if img.ndim == 3:
            pad = ((0, pad_h), (0, pad_w), (0, 0))
            img_sq = np.pad(img, pad, mode="constant", constant_values=0)
        else:
            pad = ((0, pad_h), (0, pad_w))
            img_sq = np.pad(img, pad, mode="constant", constant_values=0)
        return img_sq, (nh, nw)

    def apply_masks(self, mask: np.ndarray, old_hw: Tuple[int,int]) -> np.ndarray:
        nh, nw = old_hw
        mask_r = np.array(Image.fromarray(mask).resize((nw, nh), resample=Image.NEAREST))
        pad_h, pad_w = self.target_length - nh, self.target_length - nw
        mask_sq = np.pad(mask_r, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        return mask_sq

def scale_boxes_xyxy(boxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale [N,4] boxes in (x0,y0,x1,y1) with sx for x, sy for y."""
    out = boxes.copy()
    out[:,[0,2]] = out[:,[0,2]] * sx
    out[:,[1,3]] = out[:,[1,3]] * sy
    return out

# ------------------------------
# Simple LoRA for Linear
# ------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.A = nn.Parameter(torch.zeros((r, base.in_features)))
            self.B = nn.Parameter(torch.zeros((base.out_features, r)))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
        # freeze the base weights
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            # lora(x) = B(Ax)
            lora = F.linear(self.dropout(F.linear(x, self.A)), self.B)
            out = out + self.scaling * lora
        return out

def apply_lora_to_model(model: nn.Module, r: int, alpha: int, target_names: List[str] = None, dropout: float = 0.0, verbose=True):
    """
    Wrap nn.Linear with LoRALinear.
    If target_names is None -> inject into all Linear layers.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                if (target_names is None) or any(t in name or t in child_name for t in target_names):
                    setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                    count += 1
    if verbose:
        print(f"[LoRA] injected into {count} Linear layers (r={r}, alpha={alpha}, dropout={dropout}).")

def merge_lora_to_base(model: nn.Module):
    """
    Merge LoRA weights back to the base Linear (W <- W + scale * B @ A),
    and replace LoRALinear with the merged base Linear to keep compatibility.
    """
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear) and child.r > 0:
                base = child.base
                with torch.no_grad():
                    W = base.weight.data
                    delta = (child.B @ child.A) * child.scaling
                    W += delta
                    base.weight.data.copy_(W)
                setattr(module, child_name, base)
    return model

# ------------------------------
# Dataset (GROUPED: one image -> many masks/boxes)
# ------------------------------

def read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def read_mask(path: str) -> np.ndarray:
    m = Image.open(path).convert("L")
    m = np.array(m)
    if m.max() > 1:
        m = (m > 127).astype(np.uint8)
    return m

def mask_to_box_xyxy(m: np.ndarray) -> Tuple[int,int,int,int]:
    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0,0,0,0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)

class MappingListDatasetGrouped(Dataset):
    """
    Grouped by image: returns one sample per image containing ALL its GT masks.
    Output sample:
      {
        "image": [3,H,W] float(0..1),
        "masks": [K,1,H,W] float(0/1),
        "boxes": [K,4] float(x0,y0,x1,y1) IN THE RESIZED IMAGE SPACE,
        "K": int,
        "modality": str,
        "image_id": str,
      }
    """
    def __init__(self, base_dir: str, mapping_json: str, resize_longest_side: int = 1024,
                 bbox_from_gt: bool = True):
        super().__init__()
        self.base_dir = Path(base_dir)
        with open(self.base_dir / mapping_json, "r", encoding="utf-8") as f:
            self.items = json.load(f)  # list of {image, masks, modality?}
        self.rsz = ResizeLongestSide(resize_longest_side)
        self.bbox_from_gt = bbox_from_gt

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img_path = self.base_dir / rec["image"]
        mask_paths = [self.base_dir / p for p in rec["masks"]]

        img = read_image(str(img_path))
        H, W = img.shape[:2]

        # read each mask separately (NO UNION!)
        masks_raw: List[np.ndarray] = [read_mask(str(p)) for p in mask_paths]

        # per-mask bbox in original size
        boxes_xyxy = []
        if self.bbox_from_gt:
            for m in masks_raw:
                x0,y0,x1,y1 = mask_to_box_xyxy(m)
                if (x1-x0) < 2 or (y1-y0) < 2:
                    x0,y0,x1,y1 = 0,0,W-1,H-1
                boxes_xyxy.append([x0,y0,x1,y1])
        else:
            boxes_xyxy = [[0,0,W-1,H-1] for _ in masks_raw]
        boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32)

        # resize image once; resize masks individually; scale boxes once
        img_sq, (nh, nw) = self.rsz.apply_image(img)
        masks_sq = [self.rsz.apply_masks(m, (nh, nw)) for m in masks_raw]
        scale_x, scale_y = nw / W, nh / H
        boxes_scaled = scale_boxes_xyxy(boxes_xyxy, sx=scale_x, sy=scale_y)

        # tensors
        img_t   = torch.from_numpy(img_sq).permute(2,0,1).float() / 255.0          # [3,H,W]
        masks_t = torch.from_numpy(np.stack(masks_sq)).unsqueeze(1).float()         # [K,1,H,W]
        boxes_t = torch.from_numpy(boxes_scaled)                                    # [K,4]

        sample = {
            "image": img_t,
            "masks": masks_t,
            "boxes": boxes_t,
            "K": masks_t.shape[0],
            "modality": rec.get("modality", "_unknown_"),
            "image_id": rec["image"],
        }
        return sample

def collate_one(batch: List[Dict[str,Any]]):
    """Return the single sample directly (we enforce batch_size=1)."""
    assert len(batch) == 1, "Use batch_size=1 for grouped dataset (variable K)."
    return batch[0]

# ------------------------------
# Losses
# ------------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs*targets).sum(dim=(1,2,3)) + self.eps
        den = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + self.eps
        dice = 1 - (num/den)
        return dice.mean()

def bce_dice_loss_per_sample(logits, targets, pos_weight: float = 1.0):
    """Return per-sample BCE+Dice loss vector on dim0 ([N])."""
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
    bce = bce.mean(dim=(1,2,3))  # [N]
    probs = torch.sigmoid(logits)
    num = 2*(probs*targets).sum(dim=(1,2,3)) + 1e-6
    den = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + 1e-6
    dice = 1 - (num/den)  # [N]
    return bce + dice

# ------------------------------
# Build EdgeSAM & forward 
# ------------------------------
def _get_sam_ctor(model_type: str):
    """
    Prefer edge_sam.sam_model_registry, fallback to build_sam_vit_*.
    """
    # 1) registry
    try:
        from edge_sam import sam_model_registry
        ctor = sam_model_registry.get(model_type) or \
               sam_model_registry.get("edge_sam") or \
               sam_model_registry.get("vit_b")
        if ctor is not None:
            return ctor
    except Exception:
        pass

    # 2) fallback: build_sam_*
    try:
        from edge_sam.build_sam import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h
    except Exception as e:
        raise ImportError(f"Cannot import edge_sam / build_sam_*: {e}")

    def wrap(func):
        def _inner(checkpoint=None, **kw):
            return func(checkpoint=checkpoint, **kw)
        return _inner

    reg = {
        "edge_sam": wrap(build_sam_vit_b),
        "vit_b":    wrap(build_sam_vit_b),
        "vit_l":    wrap(build_sam_vit_l),
        "vit_h":    wrap(build_sam_vit_h),
    }
    return reg.get(model_type, reg["edge_sam"])

def build_edgesam(checkpoint_path: str, model_type: str = "edge_sam"):
    ctor = _get_sam_ctor(model_type)
    model = ctor(checkpoint=checkpoint_path)
    print(f"[Model] Loaded EdgeSAM (type={model_type})")
    return model

def _sam_preprocess(model, images: torch.Tensor):
    """
    Convert [0,1] float images to SAM's expected normalized input.
    images: [1,3,H,W]
    """
    x = images
    pm = getattr(model, "pixel_mean", None)
    ps = getattr(model, "pixel_std",  None)
    if pm is None or ps is None:
        pm = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1,3,1,1)
        ps = torch.tensor([58.395, 57.12, 57.375],   device=x.device).view(1,3,1,1)
    else:
        pm = pm.to(x.device).view(1,3,1,1); ps = ps.to(x.device).view(1,3,1,1)
    return (x*255.0 - pm) / ps

def _get_image_pe(model):
    pe_fn = getattr(model.prompt_encoder, "get_dense_pe", None)
    if callable(pe_fn):
        return pe_fn()
    pe = getattr(model.prompt_encoder, "pe", None)
    if pe is None:
        raise RuntimeError("prompt_encoder lacks dense positional embedding (neither get_dense_pe() nor pe).")
    return pe

def mask_decode(model, image_embeddings, sparse_embeddings, dense_embeddings):
    """
    Call MaskDecoder.forward with num_multimask_outputs=1.
    Handles both keyword and positional signatures.
    """
    image_pe = _get_image_pe(model)
    try:
        return model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            num_multimask_outputs=1,
        )
    except TypeError:
        return model.mask_decoder(image_embeddings, image_pe, sparse_embeddings, dense_embeddings, 1)

def forward_edgesam_multi(model: torch.nn.Module, image_3hw: torch.Tensor, boxes_k4: torch.Tensor):
    """
    Single-image encode, multi-prompt decode.
    image_3hw: [1,3,H,W]  (batch=1)
    boxes_k4:  [K,4]      (xyxy in image space)
    returns:   [K,1,H,W]
    """
    x = _sam_preprocess(model, image_3hw)              # [1,3,H,W]
    emb = model.image_encoder(x)                       # [1,C,h,w]
    H, W = image_3hw.shape[-2:]
    K = boxes_k4.shape[0]
    outs = []
    device = image_3hw.device

    for k in range(K):
        b = boxes_k4[k:k+1].to(device)
        tl, br = b[:, :2], b[:, 2:]
        boxes_pts = torch.stack([tl, br], dim=1).float()  # [1,2,2]
        try:
            sp, dn = model.prompt_encoder(points=None, boxes=boxes_pts, masks=None)
        except TypeError:
            sp, dn = model.prompt_encoder(boxes=boxes_pts)
        low = mask_decode(model, emb, sp, dn)
        low = low[0] if isinstance(low, (tuple, list)) and len(low) > 0 else low
        logit = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
        if logit.dim() == 3:      # [1,H,W]
            logit = logit.unsqueeze(1)
        elif logit.size(1) != 1:  # [1,C,H,W] -> take channel 0
            logit = logit[:, :1]
        outs.append(logit)

    return torch.cat(outs, dim=0)  # [K,1,H,W]

# ------------------------------
# L2-SP regularization 
# ------------------------------
class L2SP:
    def __init__(self, model: nn.Module, coef: float, ref_state: Dict[str,torch.Tensor]):
        self.coef = coef
        self.ref = {k: v.detach().clone() for k,v in ref_state.items()}
        self.keys = set(self.ref.keys())
        self.model = model
    def __call__(self):
        if self.coef <= 0: 
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        loss = None
        for n, p in self.model.named_parameters():
            if (p.requires_grad) and (n in self.keys):
                # FP32 + mean for stability under AMP
                diff = p.float() - self.ref[n].to(p.device).float()
                val = (diff * diff).mean()
                loss = val if loss is None else (loss + val)
        if loss is None:
            loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        return self.coef * loss

# ------------------------------
# Freeze/Unfreeze
# ------------------------------
def set_bn_eval(m: nn.Module):
    if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        m.eval()

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_last_n_blocks(model: nn.Module, n: int) -> List[str]:
    """
    Heuristic: match names containing 'block'/'layer'/'stage' etc.; unfreeze last N matches.
    Return matched tail block/module name prefixes for later use.
    """
    if n <= 0: 
        return []
    blocks = []
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ["blocks", "block", "layer", "stages", "stage"]):
            blocks.append(name)
    blocks = sorted(set(blocks))
    tail = blocks[-n:] if n <= len(blocks) else blocks
    print(f"[Unfreeze] last {n} blocks matched: {tail}")
    for name, p in model.named_parameters():
        if any(name.startswith(t) or (f".{t}." in name) for t in tail):
            p.requires_grad = True
    return tail

def enable_lora_grads(model: nn.Module, allowed_prefixes: List[str]):
    """
    Ensure LoRA A/B inside allowed scopes have gradients enabled (base weights in LoRA keep frozen).
    """
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            in_scope = False
            if any(k in name.lower() for k in ["mask_decoder", "prompt_encoder", "decoder"]):
                in_scope = True
            if any(name.startswith(t) or (f".{t}." in name) for t in allowed_prefixes):
                in_scope = True
            if in_scope:
                if m.A is not None: m.A.requires_grad = True
                if m.B is not None: m.B.requires_grad = True

# ------------------------------
# BN calibration
# ------------------------------
def calibrate_bn(model: nn.Module, loader: DataLoader, device, iters: int = 0):
    """Run no-grad forward for a few batches to refresh BN running stats. Skips if iters<=0."""
    if iters <= 0:
        return
    was_training = model.training
    model.train()
    with torch.no_grad():
        k = 0
        for sample in loader:
            img = sample["image"].to(device).unsqueeze(0)  # [1,3,H,W]
            _ = model.image_encoder(_sam_preprocess(model, img))
            k += 1
            if k >= iters:
                break
    model.train(was_training)

# ------------------------------
# Train / Val 
# ------------------------------
@torch.no_grad()
def evaluate(model, loader, device, pos_weight: float = 1.0):
    """Evaluate averaged BCE+Dice loss (with given pos_weight) and Dice across ALL masks of ALL images."""
    model.eval()
    loss_meter, dice_meter, n_masks = 0.0, 0.0, 0
    for sample in loader:
        img  = sample["image"].to(device).unsqueeze(0)   # [1,3,H,W]
        gts  = sample["masks"].to(device)                # [K,1,H,W]
        boxes= sample["boxes"].to(device)                # [K,4]

        logits = forward_edgesam_multi(model, img, boxes)  # [K,1,H,W]
        # loss/metrics per mask
        loss_vec = bce_dice_loss_per_sample(logits, gts, pos_weight=pos_weight)  # [K]
        probs = torch.sigmoid(logits)
        inter = (probs * gts).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + gts.sum(dim=(1,2,3))
        dice_vec = 2*inter / (union + 1e-6)                 # [K]

        K = gts.size(0)
        loss_meter += float(loss_vec.mean().item()) * K
        dice_meter += float(dice_vec.mean().item()) * K
        n_masks += K
    if n_masks == 0:
        return float("inf"), 0.0
    return loss_meter/n_masks, dice_meter/n_masks

def save_merged_checkpoint(model: nn.Module, path: Path):
    """Merge LoRA and save a vanilla EdgeSAM-compatible state_dict."""
    mdl = copy.deepcopy(model).cpu()
    mdl = merge_lora_to_base(mdl)
    state = mdl.state_dict()
    torch.save(state, path)
    print(f"[Checkpoint] merged+saved -> {path}")

def _epoch_path(base_path: Path, epoch: int) -> Path:
    """Turn out_path into a per-epoch filename, e.g., foo.pth -> foo_e03.pth"""
    parent = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix or ".pth"
    return parent / f"{stem}_e{epoch:02d}{suffix}"

# --- Modality helpers ---
def _canon_mod(name: str) -> str:
    return (name or "_unknown_").strip().upper().replace("-", "_")

def _parse_modality_boost(s: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    if not s: return d
    for tok in s.split(','):
        tok = tok.strip()
        if not tok: continue
        if ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        try:
            d[_canon_mod(k)] = float(v)
        except Exception:
            pass
    return d

def build_param_groups(model: nn.Module, weight_decay: float):
    """No weight decay for LoRA A/B, bias, Norm/BN; decay for the rest."""
    lora_ids = set()
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.A is not None: lora_ids.add(id(m.A))
            if m.B is not None: lora_ids.add(id(m.B))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in lora_ids or n.endswith("bias") or ("norm" in n.lower()) or ("bn" in n.lower()):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

# ------------------------------
# Train
# ------------------------------
def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Device] {device}")

    # Datasets 
    train_ds = MappingListDatasetGrouped(args.base_dir, args.train_json, args.resize_longest_side, args.bbox_from_gt)
    val_ds   = MappingListDatasetGrouped(args.base_dir, args.val_json,   args.resize_longest_side, args.bbox_from_gt)

    # DataLoaders (optional: modality-balanced sampler / fixed samples per epoch)
    sampler = None
    if args.balance_by_modality:
        mods = [_canon_mod(getattr(train_ds, 'items', [{}])[i].get('modality', '_unknown_')) for i in range(len(train_ds))]
        cnt: Dict[str, int] = {}
        for m in mods:
            cnt[m] = cnt.get(m, 0) + 1
        boost = _parse_modality_boost(args.modality_boost)
        weights = [ (boost.get(m, 1.0) / max(cnt.get(m, 1), 1)) for m in mods ]
        num_samples = args.epoch_samples if args.epoch_samples > 0 else len(weights)
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
        print(f"[Sampler] balance_by_modality on | counts={cnt} | boost={boost if boost else '{DEFAULT:1.0}'} | num_samples_per_epoch={num_samples}")
        train_loader = DataLoader(
            train_ds, batch_size=1, sampler=sampler,
            num_workers=args.workers, pin_memory=True, collate_fn=collate_one, drop_last=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed),
            persistent_workers=(args.workers > 0 and args.persistent_workers),
            prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        )
    else:
        if args.epoch_samples > 0:
            weights = [1.0] * len(train_ds)
            sampler = WeightedRandomSampler(weights, num_samples=args.epoch_samples, replacement=True)
            print(f"[Sampler] uniform WeightedRandomSampler | num_samples_per_epoch={args.epoch_samples}")
            train_loader = DataLoader(
                train_ds, batch_size=1, sampler=sampler,
                num_workers=args.workers, pin_memory=True, collate_fn=collate_one, drop_last=False,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(args.seed),
                persistent_workers=(args.workers > 0 and args.persistent_workers),
                prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=1, shuffle=True,
                num_workers=args.workers, pin_memory=True, collate_fn=collate_one, drop_last=False,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(args.seed),
                persistent_workers=(args.workers > 0 and args.persistent_workers),
                prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
            )

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_one,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=(args.workers > 0 and args.persistent_workers),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    # Small loader for BN calibration (shuffle=True, no weighted sampler)
    cal_loader = None
    if args.bn_calibrate_iters > 0:
        cal_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=collate_one,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed),
        )

    # Model
    model = build_edgesam(args.checkpoint, args.model_type)

    # L2-SP reference weights: default is current checkpoint; override with --l2sp_from if provided
    ref_state = copy.deepcopy(model.state_dict())
    if args.l2sp_from and os.path.isfile(args.l2sp_from):
        try:
            ref_state = torch.load(args.l2sp_from, map_location="cpu")
            print(f"[L2-SP] anchor loaded from {args.l2sp_from}")
        except Exception as e:
            print(f"[L2-SP] failed to load anchor from {args.l2sp_from}: {e}. Fallback to current checkpoint.")

    # Optional LoRA injection
    if args.r > 0:
        apply_lora_to_model(model, r=args.r, alpha=args.alpha, target_names=None, dropout=args.lora_dropout, verbose=True)
    else:
        print("[LoRA] disabled (r=0).")

    # Freeze all, then unfreeze head + last_n_blocks
    freeze_all(model)
    tail_blocks = unfreeze_last_n_blocks(model, args.last_n_blocks)
    for name, module in model.named_modules():
        if any(k in name.lower() for k in ["mask_decoder", "prompt_encoder", "decoder"]):
            for p in module.parameters():
                p.requires_grad = True
    # ensure LoRA A/B inside unfrozen scopes are trainable
    enable_lora_grads(model, allowed_prefixes=tail_blocks)

    # BN
    if args.freeze_bn:
        model.apply(set_bn_eval)

    model = model.to(device)

    # Optimizer (param groups: no decay for LoRA/bias/norm)
    params_groups = build_param_groups(model, weight_decay=args.weight_decay)
    trainable_params = sum(p.numel() for pg in params_groups for p in pg["params"])
    print(f"[Trainable params] {trainable_params/1e6:.2f} M")
    optimizer = torch.optim.AdamW(params_groups, lr=args.lr)
    scaler = GradScaler(enabled=args.amp)

    # L2-SP
    l2sp = None
    if args.use_l2sp > 0:
        l2sp = L2SP(model, coef=args.use_l2sp, ref_state=ref_state)
        print(f"[L2-SP] coef={args.use_l2sp}")

    best_val = float("inf")
    best_dice = 0.0
    best_path = Path(args.out_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    no_improve_epochs = 0
    patience = max(0, args.early_stop_patience)

    # Parse modality loss weights once
    modality_loss_map = _parse_modality_boost(args.modality_loss)

    def improved_by(dice_now: float, loss_now: float, dice_best: float, loss_best: float) -> bool:
        if args.select_by.lower() == "dice":
            return (dice_now > dice_best + args.dice_improve_tol) or \
                   (abs(dice_now - dice_best) <= args.dice_improve_tol and (loss_best - loss_now) > args.early_stop_min_delta)
        else:
            return ((loss_best - loss_now) > args.early_stop_min_delta) or \
                   (abs(loss_now - loss_best) <= args.early_stop_min_delta and dice_now > dice_best)

    def maybe_save(tag: str, epoch_i: int = None):
        nonlocal best_val, best_dice, no_improve_epochs
        val_loss, val_dice = evaluate(model, val_loader, device, pos_weight=args.bce_pos_weight)
        msg_tag = f"[{tag}]" if tag else ""
        print(f"{msg_tag} val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")
        if improved_by(val_dice, val_loss, best_dice, best_val):
            best_val, best_dice = val_loss, val_dice
            if args.bn_calibrate_iters > 0 and cal_loader is not None:
                calibrate_bn(model, cal_loader, device, iters=args.bn_calibrate_iters)
            save_merged_checkpoint(model, best_path)
            no_improve_epochs = 0
        else:
            # only count patience at epoch end
            pass
        # snapshot at epoch end
        if epoch_i is not None and args.save_every_epoch:
            if args.bn_calibrate_iters > 0 and cal_loader is not None:
                calibrate_bn(model, cal_loader, device, iters=args.bn_calibrate_iters)
            ep_path = _epoch_path(best_path, epoch_i)
            save_merged_checkpoint(model, ep_path)

    # Train
    global_step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        if args.freeze_bn:
            model.apply(set_bn_eval)

        epoch_loss_accum = 0.0
        seen_images = 0
        t0 = time.time()
        for sample in train_loader:
            img   = sample["image"].to(device).unsqueeze(0)  # [1,3,H,W]
            gts   = sample["masks"].to(device)               # [K,1,H,W]
            boxes = sample["boxes"].to(device)               # [K,4]
            mod   = _canon_mod(sample.get("modality", "_unknown_"))

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                logits = forward_edgesam_multi(model, img, boxes)   # [K,1,H,W]
                # Per-instance loss, then modality weighting (same weight for all K of the image)
                loss_vec = bce_dice_loss_per_sample(logits, gts, pos_weight=args.bce_pos_weight)  # [K]
                w_mod = modality_loss_map.get(mod, 1.0)
                loss_main = (loss_vec.mean()) * w_mod
            # L2-SP outside autocast in FP32
            loss = loss_main + (l2sp() if l2sp is not None else 0.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_accum += float(loss_main.item())
            seen_images += 1
            global_step += 1

            # step-wise eval (does not affect patience; only for mid-epoch checkpoints)
            if args.val_every_steps > 0 and (global_step % args.val_every_steps == 0):
                maybe_save(tag=f"step {global_step}")

        # epoch end
        dt = time.time()-t0
        trn_loss = (epoch_loss_accum / max(1, seen_images))
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={trn_loss:.4f} | time={dt:.1f}s")

        # evaluate & maybe save, count patience here
        val_loss, val_dice = evaluate(model, val_loader, device, pos_weight=args.bce_pos_weight)
        print(f"[Epoch {epoch}] val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")
        improved = improved_by(val_dice, val_loss, best_dice, best_val)
        if improved:
            best_val, best_dice = val_loss, val_dice
            if args.bn_calibrate_iters > 0 and cal_loader is not None:
                calibrate_bn(model, cal_loader, device, iters=args.bn_calibrate_iters)
            save_merged_checkpoint(model, best_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if patience > 0 and no_improve_epochs >= patience:
                print(f"[EarlyStop] No improvement for {no_improve_epochs} epochs (patience={patience}). Stop.")
                break

        # snapshot every epoch
        if args.save_every_epoch:
            if args.bn_calibrate_iters > 0 and cal_loader is not None:
                calibrate_bn(model, cal_loader, device, iters=args.bn_calibrate_iters)
            ep_path = _epoch_path(best_path, epoch)
            save_merged_checkpoint(model, ep_path)

    print(f"[Done] best val_loss={best_val:.4f}, best val_dice={best_dice:.4f}")
    if not best_path.exists():
        if args.bn_calibrate_iters > 0 and cal_loader is not None:
            calibrate_bn(model, cal_loader, device, iters=args.bn_calibrate_iters)
        save_merged_checkpoint(model, best_path)

# ------------------------------
# Args
# ------------------------------
def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_path", required=True)

    ap.add_argument("--model_type", type=str, default="edge_sam")
    ap.add_argument("--resize_longest_side", type=int, default=1024)
    ap.add_argument("--bbox_from_gt", action="store_true")

    # Finetune knobs
    ap.add_argument("--last_n_blocks", type=int, default=1)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--batch_per_image", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--use_l2sp", type=float, default=5e-4)
    ap.add_argument("--freeze_bn", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)

    # DataLoader performance/repro
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")

    # Balanced sampling / fixed epoch steps
    ap.add_argument("--balance_by_modality", action="store_true")
    ap.add_argument("--modality_boost", type=str, default="")
    ap.add_argument("--epoch_samples", type=int, default=0)

    # Saving, selection & early stop
    ap.add_argument("--save_every_epoch", action="store_true")
    ap.add_argument("--select_by", type=str, default="dice", choices=["dice","loss"])
    ap.add_argument("--dice_improve_tol", type=float, default=1e-4)
    ap.add_argument("--early_stop_patience", type=int, default=2)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-3)

    # Modality-weighted loss & BCE pos weight
    ap.add_argument("--modality_loss", type=str, default="")
    ap.add_argument("--bce_pos_weight", type=float, default=1.0)

    # L2-SP anchor override
    ap.add_argument("--l2sp_from", type=str, default="")

    # BN calibration
    ap.add_argument("--bn_calibrate_iters", type=int, default=0)

    # step-wise validation
    ap.add_argument("--val_every_steps", type=int, default=0)
    return ap.parse_args()

if __name__ == "__main__":
    args = build_args()
    if args.batch_per_image != 1:
        print("[Note] Grouped multi-prompt training enforces batch_size=1 (per-image). 'batch_per_image' is ignored.")
    train(args)
