
"""
MedSAM evaluation

Contents:
  1) Setup & globals
  2) utilities
  3) preproc
  4) model
  5) evaluation
  6) summary writer
  7) CLI
  

"""

import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import sys, time, json, glob, hashlib, cv2, torch, numpy as np
cv2.setNumThreads(0)
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from multiprocessing import get_context
from skimage.transform import resize as sk_resize
from utils.metrics import dice_score, hausdorff

# ---------------- 1) globals & determinism ----------------
PREDICT_TIMEOUT_SEC=60; DATALOADER_WORKERS=0; PREFETCH_FACTOR=2
RESTART_WORKER_EVERY_N=2000; FLUSH_EVERY=500; MODEL_LOAD_MAX_WAIT_SEC=180
TARGET=1024
try:
    torch.backends.cudnn.benchmark=False
    if hasattr(torch.backends,"cuda") and hasattr(torch.backends.cuda,"matmul"):
        torch.backends.cuda.matmul.allow_tf32=False
except: pass
CTX=get_context("spawn")

def resolve_under(root_dir, subfolder, rel):
    rel=str(rel).replace("\\","/")
    if os.path.isabs(rel) and os.path.exists(rel): return os.path.normpath(rel)
    p0=os.path.normpath(os.path.join(root_dir, rel))
    if os.path.exists(p0): return p0
    head=subfolder.lower()+"/"
    rel2=rel[len(head):] if rel.lower().startswith(head) else rel
    p1=os.path.normpath(os.path.join(root_dir, subfolder, rel2))
    if os.path.exists(p1): return p1
    return os.path.normpath(os.path.join(root_dir, subfolder, os.path.basename(rel2)))

def tight_bbox_from_mask(m):
    if m is None: return None
    u=(m>0).astype(np.uint8)
    if u.max()==0: return None
    ys,xs=np.where(u>0)
    return [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]

class GroupedDataset(Dataset):
    # Yields: (image_abs, [mask_abs], [tags])
    def __init__(self, mapping, root): self.root=root; self.items=list(mapping.items())
    def __len__(self): return len(self.items)
    def _img(self,p): return resolve_under(self.root,"images",str(p))
    def _msk(self,p): return resolve_under(self.root,"masks",str(p))
    def __getitem__(self,i):
        img_rel, mlist=self.items[i]
        img=self._img(img_rel); masks=[]; tags=[]
        for mr in mlist:
            mp=self._msk(mr); masks.append(mp)
            tags.append(os.path.splitext(os.path.basename(mp))[0].split("--")[-1])
        return img, masks, tags

def _filter_by_shard(mapping, n, i):
    if n<=1: return mapping
    out={}
    for k,v in mapping.items():
        h=int(hashlib.md5(str(k).encode()).hexdigest(),16)
        if (h % n)==i: out[str(k)]=v
    return out

# ---- strict preprocessing ----
def _rescale_to_u8(x, vmin, vmax):
    x=np.clip(x.astype(np.float32), vmin, vmax)
    if vmax<=vmin: return np.zeros_like(x, dtype=np.uint8)
    x=(x-vmin)/(vmax-vmin)*255.0
    return x.astype(np.uint8)

def _pct_u8(gray, lo=0.5, hi=99.5):
    lo_v,hi_v=np.percentile(gray.astype(np.float32), [lo,hi])
    if not np.isfinite(lo_v) or not np.isfinite(hi_v):
        lo_v,hi_v=float(gray.min()),float(gray.max())
    return _rescale_to_u8(gray, lo_v, hi_v)

def _ct_window_u8(arr, preset="soft"):
    # simple presets
    L,W={"soft":(40.0,400.0), "lung":(-160.0,1500.0), "brain":(40.0,80.0)}.get(preset, (40.0,400.0))
    return _rescale_to_u8(arr, L-W/2.0, L+W/2.0)

def _infer_modality(path):
    s=str(path).lower()
    if ("ct" in s) and ("oct" not in s): return "CT"
    if ("mr" in s) or ("mri" in s): return "MR"
    if ("xray" in s) or ("cxr" in s): return "XRAY"
    if ("ultra" in s) or ("us" in s): return "US"
    if ("mammo" in s): return "MAMMO"
    if ("oct" in s): return "OCT"
    return "RGB"

def preprocess_strict(img_path):
    # Drop alpha / expand gray
    raw=cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if raw is None: return None, None
    H0,W0=raw.shape[:2]
    if raw.ndim==2: rgb=cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.shape[2]==4: rgb=cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    else: rgb=cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # if not 8-bit visual, window/percentile to uint8
    mod=_infer_modality(img_path)
    if rgb.dtype!=np.uint8 or rgb.max()>255 or rgb.min()<0:
        if mod=="CT":
            g=cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            rgb=cv2.cvtColor(_ct_window_u8(g, "soft"), cv2.COLOR_GRAY2RGB)
        else:
            g=cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            rgb=cv2.cvtColor(_pct_u8(g,0.5,99.5), cv2.COLOR_GRAY2RGB)

    # Resize to 1024x1024 ; then per-image min-max -> [0,1]
    im=sk_resize(rgb, (TARGET,TARGET), order=3, anti_aliasing=True, preserve_range=True).astype(np.float32)
    mn, mx = float(im.min()), float(im.max())
    im = ((im - mn) / (mx - mn + 1e-6)).astype(np.float32)
    return im, (H0,W0)

class PredictorWorker:
    def __init__(self, checkpoint, model_type, timeout=PREDICT_TIMEOUT_SEC):
        self.chk=checkpoint; self.typ=model_type; self.timeout=timeout
        self.qi=None; self.qo=None; self.p=None; self._start()

    def _start(self):
        self.qi=CTX.Queue(1); self.qo=CTX.Queue(1)
        self.p=CTX.Process(target=self._worker, args=(self.qi,self.qo,self.chk,self.typ))
        self.p.daemon=True; self.p.start()
        t0=time.time()
        while time.time()-t0<MODEL_LOAD_MAX_WAIT_SEC:
            if not self.qo.empty():
                tag,payload=self.qo.get()
                if tag=="ready": return
            time.sleep(0.02)

    @staticmethod
    def _worker(qi, qo, chk, typ):
        import time as _t, torch, torch.nn.functional as F, numpy as np, cv2
        try:
            from segment_anything import sam_model_registry
        except Exception as e:
            qo.put(("ready", None))
            while True:
                job=qi.get()
                if job is None: break
                _, mps, *_=job
                qo.put(("batch", {"rows":[("err",(f"import_error:{e}",),"","")]*len(mps), "t_first":None, "t_prep":0.0}))
            return
        from __main__ import preprocess_strict, tight_bbox_from_mask, TARGET

        dev="cuda" if torch.cuda.is_available() else "cpu"
        model=sam_model_registry[typ](checkpoint=chk).to(dev).eval()
        qo.put(("ready", 1))

        while True:
            job=qi.get()
            if job is None: break
            img_path, mask_paths, tags, save_masks, masks_dir = job
            try:
                # Prep time: IO + normalize + 1024 resize + [0,1]
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0=_t.perf_counter()
                im_1024, orig=preprocess_strict(img_path)
                if im_1024 is None:
                    qo.put(("batch", {"rows":[("err",("read_image_fail",),"","")]*len(mask_paths),
                                      "t_first":None, "t_prep":0.0}))
                    continue
                H0,W0=orig
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_prep=_t.perf_counter()-t0

                # First pass latency: enc + first box decode
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0=_t.perf_counter()
                x=torch.from_numpy(im_1024.transpose(2,0,1)).unsqueeze(0).to(dev)  # 1x3x1024x1024 in [0,1]
                with torch.inference_mode(): feats=model.image_encoder(x)
                t_first=None  # set below after first decode

                rows=[]
                for j,(mp,tg) in enumerate(zip(mask_paths,tags)):
                    gt=cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    if gt is None: rows.append(("empty_gt_read_fail",(0.0,float("inf")),mp,tg)); continue
                    if gt.max()==0: rows.append(("empty_gt_all_zero",(0.0,float("inf")),mp,tg)); continue
                    # Map GT bbox → 1024 space
                    bb=tight_bbox_from_mask(gt)
                    if bb is None: rows.append(("empty_gt_all_zero",(0.0,float("inf")),mp,tg)); continue
                    x1,y1,x2,y2=bb; Hgt,Wgt=gt.shape[:2]
                    bb_1024=torch.tensor([[x1*TARGET/float(Wgt), y1*TARGET/float(Hgt),
                                           x2*TARGET/float(Wgt), y2*TARGET/float(Hgt)]],
                                         dtype=torch.float32, device=dev)
                    with torch.inference_mode():
                        sp,dn = model.prompt_encoder(points=None, boxes=bb_1024, masks=None)
                        lowres, iou = model.mask_decoder(
                            image_embeddings=feats,
                            image_pe=model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sp,
                            dense_prompt_embeddings=dn,
                            multimask_output=False
                        )
                    # First decode timing (set once)
                    if t_first is None:
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        t_first=_t.perf_counter()-t0

                    # Bilinear → original HxW; threshold 0.5
                    with torch.inference_mode():
                        logits = lowres[:,0:1]  # 1x1x256x256
                        up = torch.nn.functional.interpolate(logits, size=(H0,W0), mode="bilinear", align_corners=False)
                        pred = (up.sigmoid()>0.5).to(torch.uint8)[0,0].cpu().numpy()

                    if pred.sum()==0: rows.append(("empty_pred",(0.0,float("inf")),mp,tg)); continue

                    # Metrics
                    p255=(pred>0).astype(np.uint8)*255
                    g255=((cv2.resize((gt>0).astype(np.uint8),(W0,H0),interpolation=cv2.INTER_NEAREST))>0).astype(np.uint8)*255
                    if p255.sum()==0 or g255.sum()==0: dsc,hd=0.0,float("inf")
                    else: dsc=float(dice_score(p255,g255)); hd=float(hausdorff(p255,g255))

                    if save_masks:
                        os.makedirs(masks_dir,exist_ok=True)
                        stem=os.path.splitext(os.path.basename(img_path))[0]
                        np.save(os.path.join(masks_dir,f"{stem}--{tg}.npy"), pred.astype(np.uint8))
                    rows.append(("ok",(dsc,hd),mp,tg))

                qo.put(("batch", {"rows":rows, "t_first":t_first, "t_prep":t_prep}))
            except Exception as e:
                qo.put(("batch", {"rows":[("err",(str(e),),"","")]*len(mask_paths), "t_first":None, "t_prep":0.0}))

    def process(self, img, mps, tags, save_masks, masks_dir):
        if self.p is None or not self.p.is_alive(): self._start()
        while not self.qo.empty():
            try: self.qo.get_nowait()
            except: break
        self.qi.put((img,mps,tags,save_masks,masks_dir))
        t0=time.time()
        while time.time()-t0<self.timeout:
            if not self.qo.empty():
                tag,payload=self.qo.get()
                return payload if tag=="batch" else {"rows":[(tag,payload,"","")],"t_first":None,"t_prep":0.0}
            time.sleep(0.01)
        try:
            if self.p.is_alive(): self.p.terminate(); self.p.join(timeout=1)
        except: pass
        self._start()
        n=len(mps) if isinstance(mps,list) else 1
        return {"rows":[("timeout",(0.0,float("inf")),"","")]*n, "t_first":None, "t_prep":0.0}

    def restart(self):
        try:
            if self.p and self.p.is_alive(): self.qi.put(None); self.p.join(timeout=1)
        except: pass
        self._start()
    def close(self):
        try:
            if self.p and self.p.is_alive(): self.qi.put(None); self.p.join(timeout=1)
        except: pass

def _summarize(metrics_all_csv, n_img, n_masks, out_path):
    # Mean DSC/HD + avg prep/infer time
    import csv, numpy as np
    dscs,hds=[],[]; t_img={}; p_img={}
    if os.path.exists(metrics_all_csv):
        with open(metrics_all_csv,"r",encoding="utf-8",newline="") as f:
            r=csv.DictReader(f)
            for row in r:
                try: d=float((row.get("DSC") or row.get("dsc") or "").strip()); 
                except: d=np.nan
                if np.isfinite(d): dscs.append(d)
                hs=(row.get("HD") or row.get("hd") or "").strip().lower()
                if hs not in ("","inf","nan"):
                    try: h=float(hs); hds.append(h)
                    except: pass
                img=(row.get("image") or "").strip()
                ts=(row.get("Time") or "").strip(); ps=(row.get("Prep") or "").strip()
                try:
                    if img and img not in t_img and ts: t_img[img]=float(ts)
                except: pass
                try:
                    if img and img not in p_img and ps: p_img[img]=float(ps)
                except: pass
    mean=lambda a: float(np.mean(a)) if a else None
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("=== Evaluation Summary ===\n")
        f.write(f"Images (total): {n_img}\nMasks (total): {n_masks}\n")
        md, mh = mean(dscs), mean(hds)
        f.write(f"[box] Average DSC: {md:.4f}\n" if md is not None else "[box] Average DSC: N/A\n")
        f.write(f"[box] Average HD: {mh:.2f}\n"  if mh is not None else "[box] Average HD: N/A\n")
        at, ap = mean(list(t_img.values())), mean(list(p_img.values()))
        if at is not None: f.write(f"[box] Inference Time (enc+first decode): {at:.6f} sec/img\n")
        if ap is not None: f.write(f"[box] Prep Time (IO+preproc): {ap:.6f} sec/img\n")

def evaluate_dataset(dataset_name, dataset_root, mapping_file, checkpoint,
                     model_type="vit_b", save_masks=False,
                     num_shards=1, shard_id=0, out_name="predictions-medsam-strict"):
    worker=PredictorWorker(checkpoint, model_type, timeout=PREDICT_TIMEOUT_SEC)

    out_root=os.path.join(dataset_root,out_name); out_dir=os.path.join(out_root,f"shard{shard_id}")
    masks_dir=os.path.join(out_dir,"masks_npy"); os.makedirs(masks_dir,exist_ok=True)
    m_csv=os.path.join(out_dir,"metrics.csv"); ma_csv=os.path.join(out_dir,"metrics_all.csv")
    if not os.path.exists(m_csv): open(m_csv,"w",encoding="utf-8").write("mask,image,tag,DSC,HD,Time,Prep\n")
    if not os.path.exists(ma_csv): open(ma_csv,"w",encoding="utf-8").write("mask,image,tag,DSC,HD,reason,Time,Prep\n")

    with open(mapping_file,"r",encoding="utf-8") as f: image_to_masks=json.load(f)
    image_to_masks=_filter_by_shard(image_to_masks, num_shards, shard_id)
    ds=GroupedDataset(image_to_masks, dataset_root)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=DATALOADER_WORKERS,
                      persistent_workers=(DATALOADER_WORKERS>0),
                      prefetch_factor=(PREFETCH_FACTOR if DATALOADER_WORKERS>0 else None),
                      collate_fn=lambda xs: xs[0])

    buf_ok, buf_all=[],[]; processed=0
    for img_path, mask_paths, tags in tqdm(loader, desc="Evaluating", dynamic_ncols=True,
                                           mininterval=0.5, ascii=True,
                                           bar_format="{desc}: |{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        base=os.path.splitext(os.path.basename(img_path))[0]
        payload=worker.process(img_path, mask_paths, tags, save_masks, masks_dir)
        rows=payload.get("rows",[]); t_first=payload.get("t_first",None); t_prep=payload.get("t_prep",None)
        tstr=(f"{(t_first or 0.0):.6f}" if t_first is not None else ""); pstr=(f"{(t_prep or 0.0):.6f}" if t_prep is not None else "")
        for (status, pl, mp, tg) in rows:
            rel=os.path.relpath(mp, os.path.join(dataset_root,"masks")).replace("\\","/") if mp else f"{base}--{tg or 'NA'}"
            key=os.path.splitext(rel)[0]
            if status=="ok":
                dsc,hd=pl
                buf_ok.append(f"{key},{base},{tg},{dsc:.6f},{hd:.6f},{tstr},{pstr}\n")
                buf_all.append(f"{key},{base},{tg},{dsc:.6f},{hd:.6f},ok,{tstr},{pstr}\n")
            elif status in ("empty_pred","empty_gt_read_fail","empty_gt_all_zero","timeout"):
                buf_all.append(f"{key},{base},{tg},0.000000,inf,{status},{tstr},{pstr}\n")
            else:
                msg=pl[0] if isinstance(pl,tuple) and pl else "error"
                msg=str(msg).replace(","," ")[:200]
                buf_all.append(f"{key},{base},{tg},0.000000,inf,error:{msg},{tstr},{pstr}\n")

        processed+=len(mask_paths)
        if RESTART_WORKER_EVERY_N and processed and (processed % RESTART_WORKER_EVERY_N==0): worker.restart()
        if len(buf_ok)>=FLUSH_EVERY: open(m_csv,"a",encoding="utf-8").writelines(buf_ok); buf_ok.clear()
        if len(buf_all)>=FLUSH_EVERY: open(ma_csv,"a",encoding="utf-8").writelines(buf_all); buf_all.clear()

    if buf_ok: open(m_csv,"a",encoding="utf-8").writelines(buf_ok)
    if buf_all: open(ma_csv,"a",encoding="utf-8").writelines(buf_all)

    n_img=len(ds); n_masks=sum(len(v) for v in image_to_masks.values())
    _summarize(ma_csv, n_img, n_masks, os.path.join(out_dir,"summary.txt"))
    try: open(os.path.join(out_dir,"done.flag"),"w",encoding="utf-8").write("ok")
    except: pass
    worker.close()

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path",type=str,required=True)
    ap.add_argument("--model_type",type=str,default="vit_b")
    ap.add_argument("--base_dir",type=str,required=True)
    ap.add_argument("--sample_name",type=str,required=True)
    ap.add_argument("--mapping_file",type=str,required=True)
    ap.add_argument("--no_save",action="store_true")
    ap.add_argument("--num_shards",type=int,default=1)
    ap.add_argument("--shard_id",type=int,default=0)
    ap.add_argument("--out_name",type=str,default="predictions-medsam")
    args=ap.parse_args()

    sample_root=os.path.join(args.base_dir,args.sample_name)
    if not os.path.exists(args.mapping_file): print(f"Mapping file not found: {args.mapping_file}"); raise SystemExit(1)

    evaluate_dataset(dataset_name=args.sample_name, dataset_root=sample_root, mapping_file=args.mapping_file,
                     checkpoint=args.checkpoint_path, model_type=args.model_type,
                     save_masks=not args.no_save, num_shards=args.num_shards, shard_id=args.shard_id,
                     out_name=args.out_name)
