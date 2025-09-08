
"""

# Run and merge evaluations with external datasets

----------------------------------------------------------------------------------------



# Mapping 

$MAP300 = ".\data\External_DataSet\CVC-300\mapping.json"
$MAPCOL = ".\data\External_DataSet\CVC-ColonDB\mapping.json"
$MAPKVA = ".\data\External_DataSet\kvasir-instrument\mapping.json"
$MAPATLAS = ".\data\External_DataSet\ATLAS23_2D\mapping.json"
$MAPPH2 = ".\data\External_DataSet\PH2\mapping.json"
$MAPCHEX = ".\data\External_DataSet\Chest-X-Ray\mapping.json"
$MAPFETALH = ".\data\External_DataSet\FetalHead\mapping.json"
$MAPRIBFRAC = ".\data\External_DataSet\RibFrac\mapping.json"
$MAP300NFBS = ".\data\External_DataSet\NFBS\mapping.json"


# EdgeMedSAM
python .\evaluation\run_and_merge_external.py `
  --eval_script ".\evaluation\edgesam.py" `
  --out_name "predictions-edgemedsam" `
  --model_type edge_sam `
  --checkpoint_path ".\output_lora\edgemedsam_final.pth" `
  --base_dir ".\data\External_DataSet" `
  --sample_name "NFBS" `
  --mapping_file $MAP300NFBS `
  --num_shards 1 `
  --gpus 0 `
  --max_parallel 1 `
  --no_save npy `
  --skip_done


# SAM Vit-B
python .\evaluation\run_and_merge_external.py `
  --eval_script ".\evaluation\sam.py" `
  --out_name "predictions-sam-vit-b" `
  --checkpoint_path ".\Model\segment-anything\checkpoints\sam_vit_b.pth" `
  --model_type "vit_b" `
  --base_dir ".\data\External_DataSet" `
  --sample_name "NFBS" `
  --mapping_file $MAP300NFBS `
  --num_shards 1 `
  --gpus 0 `
  --max_parallel 1 `
  --no_save npy `
  --skip_done

# EfficientSAM Vit-T
python .\evaluation\run_and_merge_external.py `
  --eval_script ".\evaluation\efficientsam.py" `
  --out_name "predictions-efficientsam-vit-t" `
  --checkpoint_path ".\Model\EfficientSAM\efficient_sam_vitt.pt" `
  --model_type "vit_t" `
  --base_dir ".\data\External_DataSet" `
  --sample_name "NFBS" `
  --mapping_file $MAP300NFBS `
  --num_shards 1 `
  --gpus 0 `
  --max_parallel 1 `
  --no_save npy `
  --skip_done
  

# Sam for Med Vit-B
python .\evaluation\run_and_merge_external.py `
  --eval_script ".\evaluation\sam_for_med.py" `
  --out_name "predictions-sam-for-med-vit-b" `
  --checkpoint_path ".\Model\Segment-Anything-Model-for-Medical-Images\medsam_box_best_vitb.pth" `
  --model_type "vit_b" `
  --base_dir ".\data\External_DataSet" `
  --sample_name "NFBS" `
  --mapping_file $MAP300NFBS `
  --num_shards 1 `
  --gpus 0 `
  --max_parallel 1 `
  --no_save npy `
  --skip_done
  




# EdgeSAM
python .\evaluation\run_and_merge_external.py `
  --eval_script ".\evaluation\edgesam.py" `
  --out_name "predictions-edgesam" `
  --model_type edge_sam `
  --checkpoint_path ".\Model\EdgeSAM\edge_sam.pth" `
  --base_dir ".\data\External_DataSet" `
  --sample_name "NFBS" `
  --mapping_file $MAP300NFBS `
  --num_shards 1 `
  --gpus 0 `
  --max_parallel 1 `
  --no_save npy `
  --skip_done
  
----------------------------------------------------------------------------------------
"""

import os
import sys
import csv
import math
import time
import shlex
import argparse
import subprocess
from pathlib import Path
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- tools ----------------

def num(x):
    try:
        s = str(x).strip().lower()
        if s in ("", "nan"): return float("nan")
        if s == "inf": return float("inf")
        if s == "-inf": return float("-inf")
        return float(s)
    except Exception:
        return float("nan")

def read_csv_dicts(path: Path):
    rows = []
    if not path or not path.exists(): return rows
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({(k or "").strip(): v for k, v in r.items()})
    return rows

def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        for r in rows: w.writerow(r)

def get_first_key(d: dict, keys):
    for k in keys:
        for dk in (k, k.lower(), k.upper(), k.capitalize()):
            if dk in d and d[dk] != "":
                return d[dk]
    return None

def out_root(base_dir: Path, sample_name: str, out_name: str) -> Path:
    return (base_dir / sample_name / out_name)

def out_shard_dir(base_dir: Path, sample_name: str, out_name: str, shard_id: int) -> Path:
    return out_root(base_dir, sample_name, out_name) / f"shard{shard_id}"

def merged_dir(base_dir: Path, sample_name: str, out_name: str) -> Path:
    return out_root(base_dir, sample_name, out_name) / "merged"

def shard_done(out_dir: Path) -> bool:
    return (out_dir / "done.flag").exists()

# ---------------- Run Shards ----------------

def run_one_shard(eval_script: str, env_cuda: str, args, shard_id: int, passthrough_args):
    cmd = [
        sys.executable, eval_script,
        "--checkpoint_path", args.checkpoint_path,
        "--model_type", args.model_type,
        "--base_dir", args.base_dir,
        "--sample_name", args.sample_name,
        "--num_shards", str(args.num_shards),
        "--shard_id", str(shard_id),
        "--out_name", args.out_name,  
    ]
    if args.mapping_file:
        cmd += ["--mapping_file", args.mapping_file]
    if args.no_save:
        cmd += ["--no_save"]
    if args.resize_long is not None:
        cmd += ["--resize_long", str(args.resize_long)]
    if passthrough_args:
        cmd += passthrough_args

    env = os.environ.copy()
    if env_cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = env_cuda

    print(f"[RUN ] shard {shard_id}: {' '.join(cmd)}")
    out = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1)
    for line in out.stdout:
        print(f"[shard{shard_id}] {line.rstrip()}")
    return out.wait()

# ---------------- Merge Results ----------------

def merge_all(base_dir: Path, sample_name: str, out_name: str, num_shards: int):
    mdir = merged_dir(base_dir, sample_name, out_name)
    mdir.mkdir(parents=True, exist_ok=True)

    # Collect rows from all shards
    shards = [out_shard_dir(base_dir, sample_name, out_name, k) for k in range(num_shards)]
    rows_ok = []       # From metrics.csv (only ok)
    rows_all = []      # From metrics_all.csv (all results)
    images_total_sum = 0

    for sd in shards:
        rows_ok  += read_csv_dicts(sd / "metrics.csv")
        rows_all += read_csv_dicts(sd / "metrics_all.csv")
        # summary 里的 Images(total)
        sp = sd / "summary.txt"
        if sp.exists():
            with open(sp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("images (total):"):
                        try:
                            images_total_sum += int(line.split(":")[1].strip())
                        except Exception:
                            pass

    # Stats by reason
    reason_counts = {
        "ok": 0,
        "empty_pred": 0,
        "empty_gt_read_fail": 0,
        "empty_gt_all_zero": 0,
        "timeout": 0,
        "error": 0,
    }
    if rows_all:
        for r in rows_all:
            rsn = get_first_key(r, ["reason"]) or ""
            rsn = str(rsn).strip()
            if rsn == "ok":
                reason_counts["ok"] += 1
            elif rsn in reason_counts:
                reason_counts[rsn] += 1
            elif rsn.startswith("error"):
                reason_counts["error"] += 1
    else:
        def count_from_csv(path: Path):
            if not path.exists(): return 0
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    _ = next(reader, None)
                    return sum(1 for _ in reader)
            except Exception:
                return 0
        for sd in shards:
            reason_counts["empty_pred"]        += count_from_csv(sd / "empty_pred.csv")
            reason_counts["empty_gt_read_fail"]+= count_from_csv(sd / "empty_gt_read_fail.csv")
            reason_counts["empty_gt_all_zero"] += count_from_csv(sd / "empty_gt_all_zero.csv")
            reason_counts["timeout"]           += count_from_csv(sd / "timeouts.csv")
            reason_counts["error"]             += count_from_csv(sd / "errors.csv")

    # remove duplicates (same mask or same image+tag)
    def dedup(rows):
        out, seen = [], set()
        for r in rows:
            mask = get_first_key(r, ["mask"])
            image = get_first_key(r, ["image"])
            tag   = get_first_key(r, ["tag"])
            if mask and str(mask).strip():
                key = ("mask", str(mask).strip())
            else:
                key = ("imtag", (str(image).strip(), str(tag).strip()))
            if key in seen: continue
            seen.add(key)
            out.append(r)
        return out

    rows_ok  = dedup(rows_ok)
    rows_all = dedup(rows_all)

    # Union of headers 
    def header_union(rows_list):
        ks = []
        for rs in rows_list:
            for r in rs:
                for k in r.keys():
                    k1 = (k or "").strip()
                    if k1 and k1 not in ks:
                        ks.append(k1)
        prefer = ["mask","image","tag","DSC","HD","reason","Time","time_sec"]
        ordered = [k for k in prefer if k in ks] + [k for k in ks if k not in prefer]
        return ordered or prefer

    hdr_ok  = header_union([rows_ok])
    hdr_all = header_union([rows_all]) if rows_all else None

    # Write merged CSVs
    merged_metrics = mdir / "metrics.csv"
    if rows_ok:
        with open(merged_metrics, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr_ok)
            w.writeheader()
            for r in rows_ok:
                w.writerow(r)

    # Write merged all-results CSV
    if rows_all:
        with open(mdir / "metrics_all.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr_all)
            w.writeheader()
            for r in rows_all:
                w.writerow(r)

    # Summary stats 
    dscs, hds = [], []
    src_for_metric = rows_ok if rows_ok else [r for r in rows_all if (get_first_key(r, ["reason"]) or "").strip() == "ok"]
    for r in src_for_metric:
        d = num(get_first_key(r, ["DSC","dice","dsc"]))
        h = num(get_first_key(r, ["HD","hd"]))
        if math.isfinite(d): dscs.append(d)
        if math.isfinite(h): hds.append(h)

    masks_total = len(rows_all) if rows_all else len(rows_ok)
    if rows_all and reason_counts["ok"] == 0:
        ok_cnt = sum(1 for r in rows_all if (get_first_key(r, ["reason"]) or "").strip() == "ok")
        reason_counts["ok"] = ok_cnt
    if not rows_all:
        non_ok = sum(v for k,v in reason_counts.items() if k != "ok")
        if masks_total >= non_ok:
            reason_counts["ok"] = masks_total - non_ok

    # Inference time (per-image first occurrence) and throughput
    time_keys = ["Time","time_sec","t_first","inference_time"]
    time_by_image = {}
    for r in (rows_all if rows_all else rows_ok):
        img = (get_first_key(r, ["image"]) or "").strip()
        if img and img not in time_by_image:
            t = num(get_first_key(r, time_keys))
            if math.isfinite(t):
                time_by_image[img] = t

    time_list = list(time_by_image.values())
    time_mean = (sum(time_list)/len(time_list)) if time_list else None
    thrpt = (len(time_list)/sum(time_list)) if time_list and sum(time_list) > 0 else None  # img/s

    with open(mdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"=== Merged Evaluation Summary ===\n")
        f.write(f"OutName: {out_name}\n")
        f.write(f"Shards: {num_shards}\n")
        if images_total_sum > 0:
            f.write(f"Images (total): {images_total_sum}\n")
        f.write(f"Masks (total): {masks_total}\n")
        f.write(f"Average DSC: {mean(dscs):.4f}\n" if dscs else "Average DSC: N/A\n")
        f.write(f"Average HD: {mean(hds):.2f}\n"  if hds else "Average HD: N/A\n")

        if time_mean is not None:
            f.write(f"Inference Time (per_image): {time_mean:.6f} sec/img\n")
        else:
            f.write("Inference Time: N/A\n")
        if thrpt is not None:
            f.write(f"Throughput (Model-only): {thrpt:.2f} img/s\n")

        f.write("\n=== Counts by reason ===\n")
        for k in ["ok","empty_pred","empty_gt_read_fail","empty_gt_all_zero","timeout","error"]:
            f.write(f"{k}: {reason_counts.get(k, 0)}\n")

    print(f"[MERGE] Done. Merged outputs -> {mdir}")
    return str(mdir)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Unified runner for SAM-family evaluators + merger (uses --out_name).")
    ap.add_argument("--eval_script", type=str, required=True)
    ap.add_argument("--out_name", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--model_type", type=str, default="vit_b")
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--sample_name", type=str, required=True)
    ap.add_argument("--mapping_file", type=str, default=None)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--max_parallel", type=int, default=None)
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--resize_long", type=int, default=None)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--eval_extra_args", type=str, default="")
    ap.add_argument("--extra_args", type=str, default="")

    # Catch unknown args
    args, unknown = ap.parse_known_args()

    # Passthrough args
    passthrough = []
    if args.eval_extra_args:
        passthrough += shlex.split(args.eval_extra_args)
    if args.extra_args:
        passthrough += shlex.split(args.extra_args)
    if unknown:
        passthrough += [x for x in unknown if x != "--"]

    base_dir = Path(args.base_dir).resolve()
    out_root_dir = out_root(base_dir, args.sample_name, args.out_name)
    out_root_dir.mkdir(parents=True, exist_ok=True)

    # GPU list
    gpu_list = []
    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if args.max_parallel is None:
        args.max_parallel = max(1, len(gpu_list)) if gpu_list else 1

    # Tasks
    tasks = []
    for shard_id in range(args.num_shards):
        sd = out_shard_dir(base_dir, args.sample_name, args.out_name, shard_id)
        sd.mkdir(parents=True, exist_ok=True)
        if args.skip_done and shard_done(sd):
            print(f"[SKIP] shard{shard_id} already done.")
            continue
        env_cuda = gpu_list[shard_id % len(gpu_list)] if gpu_list else None
        tasks.append((env_cuda, shard_id))

    # Run shards in parallel
    if tasks:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            futs = [ex.submit(run_one_shard, args.eval_script, env_cuda, args, shard_id, passthrough)
                    for env_cuda, shard_id in tasks]
            for fut in as_completed(futs):
                rc = fut.result()
                if rc != 0:
                    sys.exit(rc)

    # Merge
    merge_all(base_dir, args.sample_name, args.out_name, args.num_shards)
    return 0

if __name__ == "__main__":
    sys.exit(main())