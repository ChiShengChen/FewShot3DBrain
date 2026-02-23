# Few-Shot Continual Learning for 3D Brain MRI (MICCAI 2026)

Frozen FOMO-60K backbone + task-specific LoRA adapters for sequential few-shot learning on infarct detection, meningioma segmentation, and brain age estimation.

## Setup

```bash
# Create venv (Python 3.10+ recommended)
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows

# Core deps (works without yucca/lightning for training)
pip install torch numpy scikit-learn tqdm

# For full FOMO preprocessing & baseline: install fomo deps
pip install -e ./fomo_baseline  # requires yucca, lightning, etc.
```

## Quick Start

**Single-task training (with dummy data):**
```bash
python run_single_task.py --task_id 1 --create_dummy --epochs 50 --patch_size 64
```

**Sequential continual learning (proposed LoRA):**
```bash
python run_continual_lora.py --create_dummy --n_shot 32 --epochs 100 --patch_size 64
```

**Baselines (linear probe, sequential FT):**
```bash
python run_baselines.py --baseline linear --create_dummy --epochs 50
python run_baselines.py --baseline sequential_ft --create_dummy --epochs 50
```

## Data

- **Real data:** Run FOMO preprocessing first:
  - Preprocess task data: `python fomo_baseline/src/data/preprocess/run_preprocessing.py --taskid=1 --source_path=/path/to/raw`
  - Place preprocessed data in `data/preprocessed/Task001_FOMO1/` etc.
- **Dummy data:** Use `--create_dummy` to generate minimal test data.

### Plan B: Public Datasets（FOMO 下游資料不可用時）

若 FOMO 官方下游評估資料無法取得，可使用公開資料集替代。執行：

```bash
# 依賴：pip install nibabel scipy pandas openpyxl xlrd

# 自動下載並轉換 Task 1 (ISLES)、Task 3 (IXI)
python scripts/prepare_plan_b_data.py --all --download

# 建立 few-shot 子集 (16/32/64) 及 train/val split
python scripts/prepare_plan_b_data.py --create_fewshot --n_shots 16 32 64

# Task 2 需手動從 Synapse 下載 (需帳號 + DUA)
# https://www.synapse.org/#!Synapse:syn51514105
python scripts/prepare_plan_b_data.py --task 2 --source_dir /path/to/BraTS-Meningioma

# 其他
python scripts/prepare_plan_b_data.py --task 1 --source_dir /path/to/ISLES-2022
python scripts/prepare_plan_b_data.py --task 3 --source_dir /path/to/IXI
```

| Task | 資料集 | 說明 |
|------|--------|------|
| 1 | ISLES 2022 (Zenodo 7153326) | 腦梗塞分割，FLAIR/DWI/ADC → 4ch（補零）→ 分類標籤，z-score 正規化 |
| 2 | BraTS Meningioma / Glioma | 腦瘤分割，T2/FLAIR/T1ce → 3ch |
| 3 | IXI | 腦齡回歸，T1/T2 + 年齡（IXI.xls）|

輸出格式與 `FewShotFOMODataset` 相容。`--create_fewshot` 會產生 `Task*_fewshot16/32/64/` 子目錄及 `splits_*.json`。

### Task 3 (IXI) 預處理注意事項

IXI NIfTI 檔案的 affine 對角線**不代表 voxel spacing**；若誤用 affine 取得 spacing，resample 會產生異常形狀（如維度為 0）。本專案已改為使用 `nibabel` 的 `header.get_zooms()` 取得正確 spacing。

若之前曾預處理過 Task 3 且產生形狀異常（例如 `(2, 0, 30, 0)`），請重新執行：

```bash
python scripts/prepare_plan_b_data.py --task 3 --source_dir data/plan_b_raw/ixi
```

預處理完成後，可確認樣本形狀應為 `(2, D, H, W)`，其中 D,H,W ≥ 64（例如約 `(2, 240, 240, 180)`）。

## Pretrained Weights（強烈推薦冠軍團隊權重）

**FOMO25 官方 baseline 權重**（HuggingFace）現已不可用。請改用 **冠軍團隊（jbanusco）開源權重**：

| 項目 | 說明 |
|------|------|
| **來源** | [jbanusco/fomo25](https://github.com/jbanusco/fomo25) - FOMO25 方法賽道第 1 名 |
| **架構** | MultiModalUNetVAE（U-Net CNN，體積約為官方 ViT baseline 的 1/10） |
| **單卡 3090** | 適合在單張 3090 上進行下游微調 |

### 下載方式一：直接下載權重檔（推薦）

```bash
# 使用專案腳本
chmod +x scripts/download_champion_weights.sh
./scripts/download_champion_weights.sh

# 或手動下載
mkdir -p weights
curl -L -o weights/fomo25_mmunetvae_pretrained.ckpt \
  https://github.com/jbanusco/fomo25/releases/download/v1.0.0/fomo25_mmunetvae_pretrained.ckpt
```

### 下載方式二：Docker 映像（內含權重）

```bash
docker pull jbanusco/sslmmunetave:1.0.0
```

### 使用方式

目前本專案 backbone 為 fomo_baseline 的 `unet_b`；冠軍權重對應架構為 `mmunetvae`。若要完整使用冠軍權重，需：
- 選項 A：將 [jbanusco/fomo25](https://github.com/jbanusco/fomo25) 作為 backbone 整合（含 `mmunetvae`）
- 選項 B：以無權重方式訓練，或等待官方 baseline 權重恢復

**使用冠軍權重（需整合 mmunetvae）**：執行 `python run_continual_champion.py`（見下方）。

## Project Structure

```
FewShot3DBrain/
├── src/                 # Core: data.py, lora.py, models.py, train.py, backbone, replay
├── scripts/             # prepare_plan_b_data.py, run_miccai_experiments, download_champion_weights
├── run_*.py             # Entry points: single_task, continual_lora, baselines, ablations, multi_seed
├── fomo_baseline/       # FOMO25 baseline (cloned)
├── champion_fomo/       # jbanusco/fomo25 (1st-place)
├── data/                # preprocessed/, plan_b_raw/ (gitignored)
├── outputs/             # Experiment outputs (gitignored)
├── weights/             # Pretrained weights (gitignored)
├── MICCAI_paper/        # LaTeX paper
└── README.md, EXPERIMENT_RESULTS.md, MICCAI2026_PAPER_PLAN.md
```
