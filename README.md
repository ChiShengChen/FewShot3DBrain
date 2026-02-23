# Few-Shot Continual Learning for 3D Brain MRI

<p align="center">
  <a href="#english">English</a> · <a href="#%E4%B8%AD%E6%96%87">中文</a>
</p>

---

<a name="english"></a>

## English

Frozen FOMO-60K backbone + task-specific LoRA adapters for sequential few-shot learning on infarct detection, meningioma segmentation, and brain age estimation.

### Setup

```bash
# Create venv (Python 3.10+ recommended)
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows

# Core deps (works without yucca/lightning for training)
pip install torch numpy scikit-learn tqdm

# For full FOMO preprocessing & baseline: install fomo deps
pip install -e ./fomo_baseline  # requires yucca, lightning, etc.
```

### Quick Start

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

### Data

- **Real data:** Run FOMO preprocessing first:
  - Preprocess task data: `python fomo_baseline/src/data/preprocess/run_preprocessing.py --taskid=1 --source_path=/path/to/raw`
  - Place preprocessed data in `data/preprocessed/Task001_FOMO1/` etc.
- **Dummy data:** Use `--create_dummy` to generate minimal test data.

#### Plan B: Public Datasets (when FOMO downstream data is unavailable)

```bash
# Dependencies: pip install nibabel scipy pandas openpyxl xlrd

# Auto-download and convert Task 1 (ISLES), Task 3 (IXI)
python scripts/prepare_plan_b_data.py --all --download

# Create few-shot subsets (16/32/64) and train/val split
python scripts/prepare_plan_b_data.py --create_fewshot --n_shots 16 32 64

# Task 2 requires manual download from Synapse (account + DUA)
# https://www.synapse.org/#!Synapse:syn51514105
python scripts/prepare_plan_b_data.py --task 2 --source_dir /path/to/BraTS-Meningioma

# Others
python scripts/prepare_plan_b_data.py --task 1 --source_dir /path/to/ISLES-2022
python scripts/prepare_plan_b_data.py --task 3 --source_dir /path/to/IXI
```

| Task | Dataset | Description |
|------|---------|-------------|
| 1 | ISLES 2022 (Zenodo 7153326) | Infarct seg: FLAIR/DWI/ADC → 4ch (pad) → class label, z-score norm |
| 2 | BraTS Meningioma / Glioma | Tumour seg: T2/FLAIR/T1ce → 3ch |
| 3 | IXI | Brain age reg: T1/T2 + age (IXI.xls) |

Output format is compatible with `FewShotFOMODataset`. `--create_fewshot` produces `Task*_fewshot16/32/64/` subdirs and `splits_*.json`.

#### Task 3 (IXI) Preprocessing Notes

IXI NIfTI affine diagonal **does not represent voxel spacing**; using affine for spacing can produce bad shapes (e.g. dim 0). This project uses `nibabel` `header.get_zooms()` instead.

If Task 3 was preprocessed and produced bad shapes (e.g. `(2, 0, 30, 0)`), re-run:

```bash
python scripts/prepare_plan_b_data.py --task 3 --source_dir data/plan_b_raw/ixi
```

Expected shape after preprocessing: `(2, D, H, W)` with D,H,W ≥ 64.

### Pretrained Weights (Champion team weights recommended)

**FOMO25 official baseline weights** (HuggingFace) are no longer available. Use **champion team (jbanusco) open-source weights**:

| Item | Description |
|------|-------------|
| **Source** | [jbanusco/fomo25](https://github.com/jbanusco/fomo25) - FOMO25 method track 1st place |
| **Architecture** | MultiModalUNetVAE (U-Net CNN, ~1/10 size of official ViT baseline) |
| **Single 3090** | Suitable for downstream fine-tuning |

**Download:**
```bash
chmod +x scripts/download_champion_weights.sh
./scripts/download_champion_weights.sh
```

**Usage:** Current backbone is fomo_baseline `unet_b`; champion weights use `mmunetvae`. To use champion weights fully: integrate [jbanusco/fomo25](https://github.com/jbanusco/fomo25) (option A) or train without pretrained weights (option B). Run `python run_continual_champion.py` when integrated.

### Project Structure

```
FewShot3DBrain/
├── src/                 # Core: data.py, lora.py, models.py, train.py, backbone, replay
├── scripts/             # prepare_plan_b_data.py, run_experiments, download_champion_weights
├── run_*.py             # Entry points: single_task, continual_lora, baselines, ablations, multi_seed
├── fomo_baseline/       # FOMO25 baseline (cloned)
├── champion_fomo/       # jbanusco/fomo25 (1st-place)
├── data/                # preprocessed/, plan_b_raw/ (gitignored)
├── outputs/             # Experiment outputs (gitignored)
├── weights/             # Pretrained weights (gitignored)
├── paper/               # LaTeX paper
└── README.md, EXPERIMENT_RESULTS.md, PAPER_PLAN.md
```

---

<a name="中文"></a>

## 中文

凍結 FOMO-60K backbone + 任務專用 LoRA adapters，用於腦梗塞檢測、腦膜瘤分割、腦齡估計的序貫少樣本學習。

### 安裝

```bash
# 建議 Python 3.10+
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 核心依賴（可不裝 yucca/lightning 進行訓練）
pip install torch numpy scikit-learn tqdm

# 完整 FOMO 預處理與 baseline
pip install -e ./fomo_baseline  # 需 yucca, lightning 等
```

### 快速開始

**單任務訓練（dummy 資料）：**
```bash
python run_single_task.py --task_id 1 --create_dummy --epochs 50 --patch_size 64
```

**序貫持續學習（LoRA）：**
```bash
python run_continual_lora.py --create_dummy --n_shot 32 --epochs 100 --patch_size 64
```

**Baseline（linear probe、sequential FT）：**
```bash
python run_baselines.py --baseline linear --create_dummy --epochs 50
python run_baselines.py --baseline sequential_ft --create_dummy --epochs 50
```

### 資料

- **真實資料：** 先執行 FOMO 預處理，將結果放至 `data/preprocessed/Task001_FOMO1/` 等。
- **Dummy 資料：** 使用 `--create_dummy` 產生測試用資料。

#### Plan B：公開資料集（FOMO 下游資料不可用時）

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

#### Task 3 (IXI) 預處理注意事項

IXI NIfTI 的 affine 對角線**不代表 voxel spacing**；誤用 affine 取得 spacing 會導致 resample 異常（如維度為 0）。本專案改用 `nibabel` 的 `header.get_zooms()` 取得正確 spacing。

若 Task 3 曾預處理且產生異常形狀（如 `(2, 0, 30, 0)`），請重新執行：

```bash
python scripts/prepare_plan_b_data.py --task 3 --source_dir data/plan_b_raw/ixi
```

預處理後樣本形狀應為 `(2, D, H, W)`，D,H,W ≥ 64。

### 預訓練權重（強烈推薦冠軍團隊權重）

**FOMO25 官方 baseline 權重**（HuggingFace）現已不可用。請改用 **冠軍團隊（jbanusco）開源權重**：

| 項目 | 說明 |
|------|------|
| **來源** | [jbanusco/fomo25](https://github.com/jbanusco/fomo25) - FOMO25 方法賽道第 1 名 |
| **架構** | MultiModalUNetVAE（U-Net CNN，體積約為官方 ViT baseline 的 1/10） |
| **單卡 3090** | 適合在單張 3090 上進行下游微調 |

**下載：**
```bash
chmod +x scripts/download_champion_weights.sh
./scripts/download_champion_weights.sh
```

**使用方式：** 目前 backbone 為 fomo_baseline 的 `unet_b`；冠軍權重對應 `mmunetvae`。若要完整使用：整合 [jbanusco/fomo25](https://github.com/jbanusco/fomo25)（選項 A）或改為無權重訓練（選項 B）。整合完成後執行 `python run_continual_champion.py`。

### 專案結構

```
FewShot3DBrain/
├── src/                 # 核心：data.py, lora.py, models.py, train.py, backbone, replay
├── scripts/             # prepare_plan_b_data.py, run_experiments, download_champion_weights
├── run_*.py             # 入口：single_task, continual_lora, baselines, ablations, multi_seed
├── fomo_baseline/       # FOMO25 baseline（已 clone）
├── champion_fomo/       # jbanusco/fomo25（第 1 名）
├── data/                # preprocessed/, plan_b_raw/（gitignored）
├── outputs/             # 實驗輸出（gitignored）
├── weights/             # 預訓練權重（gitignored）
├── paper/               # LaTeX 論文
└── README.md, EXPERIMENT_RESULTS.md, MICCAI2026_PAPER_PLAN.md
```
