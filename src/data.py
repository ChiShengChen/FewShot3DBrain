"""
Few-shot data loading for FOMO tasks.
Works with preprocessed data (numpy .npy + .txt labels).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Literal
from pathlib import Path


# Task configs (from FOMO task_configs)
TASK_CONFIGS = {
    1: {
        "name": "Task001_FOMO1",
        "task_type": "classification",
        "num_classes": 2,
        "modalities": 4,  # DWI, T2FLAIR, ADC, SWI_OR_T2STAR
    },
    2: {
        "name": "Task002_FOMO2",
        "task_type": "segmentation",
        "num_classes": 2,
        "modalities": 3,  # DWI, T2FLAIR, SWI_OR_T2STAR
    },
    3: {
        "name": "Task003_FOMO3",
        "task_type": "regression",
        "num_classes": 1,
        "modalities": 2,  # T1, T2
    },
}


def list_case_files(data_dir: str, task_id: int) -> List[str]:
    """List all case files (without .npy suffix) in task directory."""
    task_cfg = TASK_CONFIGS[task_id]
    task_name = task_cfg["name"]
    task_dir = os.path.join(data_dir, task_name)
    if not os.path.isdir(task_dir):
        return []
    files = []
    for f in os.listdir(task_dir):
        if f.endswith(".npy") and not f.endswith(".pkl") and not f.endswith("_seg.npy"):
            case_id = f[:-4]  # strip .npy
            files.append(os.path.join(task_dir, case_id))
    return sorted(set(files))


def few_shot_split(
    files: List[str],
    n_shot: int,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split files into train (few-shot) and val.
    n_shot: number of training samples (16, 32, or 64).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    files = [files[i] for i in perm]
    
    n_train = min(n_shot, len(files))
    n_val = max(0, int((len(files) - n_train) * val_ratio))
    
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val] if n_val > 0 else files[n_train : n_train + 1]
    
    return train_files, val_files


class FewShotFOMODataset(Dataset):
    """
    Dataset for few-shot FOMO tasks.
    Loads preprocessed .npy volumes and .txt labels.
    """
    def __init__(
        self,
        files: List[str],
        patch_size: Tuple[int, int, int],
        task_type: Literal["classification", "segmentation", "regression"],
        transform=None,
        target_patch_size_for_crop: Optional[Tuple[int, int, int]] = None,
    ):
        self.files = files
        self.patch_size = patch_size
        self.task_type = task_type
        self.transform = transform
        self.crop_size = target_patch_size_for_crop or patch_size

    def __len__(self) -> int:
        return len(self.files)

    def _load_volume(self, path: str) -> np.ndarray:
        npy_path = path + ".npy" if not path.endswith(".npy") else path
        vol = np.load(npy_path, allow_pickle=True)
        if vol.ndim == 3:
            vol = vol[np.newaxis, ...]  # add channel
        return vol.astype(np.float32)

    def _load_label(self, path: str) -> np.ndarray:
        base = path.replace(".npy", "").rstrip(".txt")
        txt_path = base + ".txt"
        if self.task_type == "classification":
            lbl = np.loadtxt(txt_path, dtype=np.int64) if os.path.exists(txt_path) else np.array(0)
        elif self.task_type == "regression":
            lbl = np.loadtxt(txt_path, dtype=np.float32) if os.path.exists(txt_path) else np.array(50.0)
            lbl = np.atleast_1d(lbl)
        else:
            # segmentation: label is same base path with _seg suffix
            seg_path = base + "_seg.npy"
            if os.path.exists(seg_path):
                lbl = np.load(seg_path, allow_pickle=True)
                if lbl.ndim == 3:
                    lbl = lbl[np.newaxis, ...]  # (D,H,W) -> (1,D,H,W)
                elif lbl.shape[0] > 1:
                    lbl = lbl[:1]  # keep single channel for binary seg
            else:
                lbl = np.zeros((1,) + self._load_volume(path).shape[1:], dtype=np.float32)
        return lbl

    def _random_crop(self, vol: np.ndarray, label: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Random crop to patch_size."""
        c, d, h, w = vol.shape
        pd, ph, pw = self.crop_size
        if d < pd or h < ph or w < pw:
            # Pad
            vol = np.pad(vol, ((0, 0), (0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w))), mode="constant", constant_values=0)
            if label is not None and label.ndim == 4:
                label = np.pad(label, ((0, 0), (0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w))), mode="constant", constant_values=0)
            d, h, w = vol.shape[1:]
        sd = np.random.randint(0, d - pd + 1) if d > pd else 0
        sh = np.random.randint(0, h - ph + 1) if h > ph else 0
        sw = np.random.randint(0, w - pw + 1) if w > pw else 0
        vol = vol[:, sd : sd + pd, sh : sh + ph, sw : sw + pw]
        if label is not None and label.ndim == 4:
            label = label[:, sd : sd + pd, sh : sh + ph, sw : sw + pw]
        return vol, label

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        vol = self._load_volume(path)
        label = self._load_label(path)
        
        if self.task_type == "segmentation" and isinstance(label, np.ndarray) and label.ndim == 3:
            label = label[np.newaxis, ...]
        
        vol, label = self._random_crop(vol, label if self.task_type == "segmentation" else None)
        
        if self.task_type == "segmentation":
            if label is None:
                label = np.zeros((1,) + vol.shape[1:], dtype=np.float32)
            label = np.asarray(label, dtype=np.float32)
            if label.ndim == 3:
                label = label[np.newaxis, ...]
            elif label.shape[0] == vol.shape[0]:  # mistakenly loaded volume
                label = np.zeros((1,) + vol.shape[1:], dtype=np.float32)
            else:
                label = label[:1]
        
        if self.task_type == "classification":
            if label is None:
                label = 0
            label = int(label) if np.isscalar(label) else int(np.asarray(label).flatten()[0])
        elif self.task_type == "regression":
            if label is None:
                label = 50.0
            label = float(np.asarray(label).flatten()[0])
        
        data = {
            "image": torch.from_numpy(vol).float(),
            "label": torch.tensor(label, dtype=torch.long if self.task_type == "classification" else torch.float32),
        }
        if self.transform:
            data = self.transform(data)
        return data


def _seg_collate(batch):
    """Custom collate for segmentation: ensure label is (B,1,D,H,W)."""
    images = torch.stack([b["image"] for b in batch])
    labels = []
    for b in batch:
        l = b["label"]
        if l.dim() == 3:
            l = l.unsqueeze(0)
        if l.shape[0] > 1:
            l = l[:1]
        labels.append(l)
    labels = torch.stack(labels)
    return {"image": images, "label": labels}


def get_few_shot_dataloaders(
    data_dir: str,
    task_id: int,
    n_shot: int,
    patch_size: Tuple[int, int, int],
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val dataloaders for few-shot setting.
    """
    cfg = TASK_CONFIGS[task_id]
    files = list_case_files(data_dir, task_id)
    if len(files) == 0:
        raise FileNotFoundError(f"No data found in {data_dir} for task {task_id}. Run preprocessing first.")
    
    train_files, val_files = few_shot_split(files, n_shot=n_shot, seed=seed)
    
    train_ds = FewShotFOMODataset(
        train_files,
        patch_size=patch_size,
        task_type=cfg["task_type"],
    )
    val_ds = FewShotFOMODataset(
        val_files,
        patch_size=patch_size,
        task_type=cfg["task_type"],
    )
    
    collate_fn = _seg_collate if cfg["task_type"] == "segmentation" else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


def create_dummy_data(out_dir: str, task_id: int = 1, n_samples: int = 32):
    """
    Create minimal dummy data for testing when real FOMO data is not available.
    """
    cfg = TASK_CONFIGS[task_id]
    task_dir = os.path.join(out_dir, cfg["name"])
    os.makedirs(task_dir, exist_ok=True)
    
    n_mod = cfg["modalities"]
    shape = (n_mod, 64, 64, 64)  # small dummy volume
    
    for i in range(n_samples):
        base = os.path.join(task_dir, f"sample_{i:04d}")
        vol = np.random.randn(*shape).astype(np.float32) * 0.5
        np.save(base + ".npy", vol)
        
        if cfg["task_type"] == "classification":
            lbl = np.random.randint(0, cfg["num_classes"])
        elif cfg["task_type"] == "regression":
            lbl = np.array(40.0 + np.random.randn() * 10, dtype=np.float32)
        else:
            lbl = np.zeros((1,) + shape[1:], dtype=np.float32)
            lbl[0, 20:40, 20:40, 20:40] = 1  # dummy mask
        
        if cfg["task_type"] == "segmentation":
            np.save(base + "_seg.npy", lbl)
        else:
            np.savetxt(base + ".txt", np.atleast_1d(lbl), fmt="%f" if cfg["task_type"] == "regression" else "%d")
    
    print(f"Created {n_samples} dummy samples in {task_dir}")
