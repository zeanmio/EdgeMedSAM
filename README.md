# Efficient SAM-based Models for Medical Image Segmentation (EdgeMedSAM)

---

## 0) Overview
This repository evaluates SAM-family models on medical images under a deterministic, unified tight-box protocol, and introduces **EdgeMedSAM**, a lightweight fine-tuned variant designed to preserve runtime efficiency.

**Evaluation** focuses on reproducibility and steady-state timing under a single, comparable protocol, producing accuracy and efficiency summaries across modalities.

**Fine-tuning** applies parameter-efficient adaptation to EdgeSAM and exports a deployment-ready checkpoint without changing the inference path.

---

## 1) Repository layout
```
EdgeMedSAM/
├─ evaluation/                  
│  ├─ edgesam.py, efficientsam.py, sam.py, sam_for_med.py, ...  # Model entry points
│  ├─ run_and_merge_generic/balance/external.py  # Runner / sharding & merge
│  └─ utils/                    # Shared helpers
├─ training/
│  └─ train_edgemedsam.py       # EdgeSAM + LoRA training > EdgeMedSAM
├─ output_lora/                 # LoRA checkpoints / merged exports
├─ model/                       # Base models (not in this repo)
├─ data/                        # Datasets + mapping.json (not in this repo)
├─ requirements-sam-env.txt     # General env (for SAM/SAM-for-Med/EfficientSAM/MobileSAM/SAM-Med-2D)
├─ requirements-sam-py311.txt   # Python 3.11 env (for EdgeSAM/MedSAM/Lite-MedSAM)
└─ requirements-sam-fast.txt    # Lightweight/fast eval env (for FastSAM)
```
---

## 2) Environments
**PowerShell (Conda):**
```powershell
conda create -n sam-fast python=3.11 -y
conda activate sam-fast
pip install -r requirements-sam-fast.txt
# or: pip install -r requirements-sam-env.txt / requirements-sam-py311.txt
```
**Bash:**
```bash
conda create -n sam-fast python=3.11 -y && conda activate sam-fast
pip install -r requirements-sam-fast.txt
```
> Install `torch/torchvision` wheels that match your CUDA version.

**Export/refresh the env:**
```powershell
conda activate sam-fast
pip freeze > requirements-sam-fast.txt
```

---

## 3) Data format
- **Images**: 8-bit RGB (uint8).
- **Masks**: single-channel binary (0/255).
- **Box prompt**: derived from the GT mask bounding rectangle.
- **Shared protocol**: all models read the same mapping.json to ensure fairness and reproducibility.

Both the random and balanced datasets for evaluation were extracted from the original SA-Med2D-20Mv1 dataset, uniformly using random number seed 1. The training dataset extracted all images and corresponding masks from the SA-Med2D-20Mv1 dataset except for the CT/MR modalities, while the CT/MR modalities were randomly sampled with random number seed 1. Based on these datasets, the evaluation and training results can be reproduced.

---

## 4) Evaluation (unified protocol) 

```
All commands are commented above the evaluation runner and trainer. All commands can be run from the base directory.

```

**EdgeSAM (base) example:**
```powershell
python .\evaluation\run_and_merge_generic.py `
  --eval_script       .\evaluation\edgesam.py `
  --out_name          predictions-edgesam `
  --model_type        edge_sam `
  --checkpoint_path   .\Model\EdgeSAM\edge_sam.pth `
  --base_dir          .\data\mix_50k_seed1 `
  --mapping_file      .\data\mix_50k_seed1\split_811\test_map_for_eval.json `
  --num_shards 1 --gpus 0 --max_parallel 1 `
  --no_save npy --skip_done
```

**EdgeMedSAM (LoRA) example:**
```powershell
python .\evaluation\run_and_merge_generic.py `
  --eval_script       .\evaluation\edgesam.py `
  --out_name          predictions-edgemedsam `
  --model_type        edge_sam `
  --checkpoint_path   .\output_lora\edgemedsam_final.pth `
  --base_dir          .\data\mix_50k_seed1 `
  --mapping_file      .\data\mix_50k_seed1\split_811\test_map_for_eval.json `
  --num_shards 1 --gpus 0 --max_parallel 1 `
  --no_save npy --skip_done
```
> `edgesam.py` merges LoRA into the base at load/inference time.

**External suite (single dataset example):**
```powershell
$MAPFETALH = ".\data\External_DataSet\FetalHead\mapping.json"
python .\evaluation\run_and_merge_external.py `
  --eval_script     .\evaluation\edgesam.py `
  --out_name        predictions-edgemedsam-1 `
  --model_type      edge_sam `
  --checkpoint_path .\output_lora\edgemedsam_final.pth `
  --base_dir        .\data\External_DataSet `
  --sample_name     FetalHead `
  --mapping_file    $MAPFETALH `
  --num_shards 1 --gpus 0 --max_parallel 1 `
  --no_save npy --skip_done
```

**Outputs**: per-shard results are merged into `predictions-*` folders. You can aggregate CSVs for arithmetic or weighted means (see §6).

---

## 5) Fine-tuning (EdgeSAM + LoRA → EdgeMedSAM)

Show help:
```powershell
python .\train_edgemedsam.py -h
```

Typical command (adjust names to match your script args):
```powershell
python .\training\train_edgemedsam.py `
  --base_checkpoint  .\Model\EdgeSAM\edge_sam.pth `
  --train_map        .\data\mix_50k_seed1\split_811\train_map.json `
  --val_map          .\data\mix_50k_seed1\split_811\val_map.json `
  --img_root         .\data\mix_50k_seed1 `
  --save_dir         .\output_lora `
  --lora_rank        32 `
  --epochs           50000 `
  --batch_size       32 `
  --lr               1e-4 `
  --precision        fp16 `
  --log_every        200 `
  --save_every       2000
```
The produced LoRA checkpoint lives under `output_lora/` and is merged at inference time by `edgesam.py`. If you expose an explicit *merged-export* flag in your code, you can also create a single consolidated checkpoint for deployment.

---

## 6) Repro & FAQ
- Fix seeds for numpy/torch/CuDNN; keep the same `mapping.json` across models.  
- Throughput is reported as **imgs/s**; inference time as **s/img** (exclude I/O).  
- `torch.cuda.is_available()==False`: verify CUDA driver & wheel compatibility.  
- OOM: reduce `--batch_size` or use `--precision fp16`.


## 7) Acknowledgments
This project evaluates/adapts several public models and implementations: **SAM**, **SAM-for-Med**, **MedSAM**, **MedSAM_Lite**, **SAM-Med2D**, **EdgeSAM**, **EfficientSAM**, **MobileSAM**, **FastSAM**, etc. Names belong to their respective authors/organisations.
