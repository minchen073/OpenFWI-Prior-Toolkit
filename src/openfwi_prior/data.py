"""Memory-mapped OpenFWI velocity fields and dataset splits."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VelocityDataset(Dataset):
    """Load individual (1, 70, 70) fields in normalized velocity units."""

    def __init__(
        self, root, files, samples_per_file=500, center_m_s=3000.0, scale_m_s=1500.0
    ):
        self.root = Path(root).expanduser().resolve()
        self.paths = [(self.root / name).resolve() for name in files]
        self.samples_per_file = int(samples_per_file)
        self.center_m_s = float(center_m_s)
        self.scale_m_s = float(scale_m_s)
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("Dataset files must be nonempty and unique")
        if self.samples_per_file <= 0 or self.scale_m_s <= 0:
            raise ValueError("samples_per_file and scale_m_s must be positive")
        self.arrays = []
        for path in self.paths:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 4 or array.shape[1:] != (1, 70, 70):
                raise ValueError(f"Expected (N, 1, 70, 70): {path}, got {array.shape}")
            if array.shape[0] < self.samples_per_file:
                raise ValueError(f"Not enough samples in {path}")
            self.arrays.append(array)

    def __len__(self):
        return len(self.arrays) * self.samples_per_file

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index, sample_index = divmod(index, self.samples_per_file)
        raw = np.array(
            self.arrays[file_index][sample_index], dtype=np.float32, copy=True
        )
        return torch.from_numpy((raw - self.center_m_s) / self.scale_m_s)

    def describe(self):
        return {
            "root": str(self.root),
            "files": [p.name for p in self.paths],
            "num_files": len(self.paths),
            "samples_per_file": self.samples_per_file,
            "length": len(self),
            "shape": [1, 70, 70],
            "center_m_s": self.center_m_s,
            "scale_m_s": self.scale_m_s,
        }


def make_datasets(config):
    root = Path(config["root"]).expanduser().resolve()
    train_paths = {(root / name).resolve() for name in config["train_files"]}
    val_paths = {(root / name).resolve() for name in config["validation_files"]}
    if train_paths & val_paths:
        raise ValueError("Training and monitoring files must be disjoint")
    kwargs = {
        key: config[key] for key in ("samples_per_file", "center_m_s", "scale_m_s")
    }
    return (
        VelocityDataset(root, config["train_files"], **kwargs),
        VelocityDataset(root, config["validation_files"], **kwargs),
    )
