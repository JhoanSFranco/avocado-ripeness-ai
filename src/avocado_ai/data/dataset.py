"""PyTorch dataset over the prepared metadata."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from avocado_ai.data.metadata import TARGET_COLUMN


class AvocadoImageDataset(Dataset):
    """Yields `(image, class_index, standardized_target)` tuples.

    Args:
        frame: Rows from `metadata.csv` (must contain `image_path`, `stage`
            and the regression target).
        image_root: Folder that `image_path` is relative to.
        transform: Image transform (see `avocado_ai.ml.transforms`).
        target_mean / target_std: Standardization statistics computed on the
            *training* split only.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
        target_mean: float,
        target_std: float,
    ) -> None:
        frame = frame.dropna(subset=["image_path", "stage", TARGET_COLUMN])
        self.paths = [Path(image_root) / p for p in frame["image_path"]]
        self.labels = torch.tensor(frame["stage"].to_numpy() - 1, dtype=torch.long)
        self.targets = torch.tensor(
            (frame[TARGET_COLUMN].to_numpy() - target_mean) / target_std, dtype=torch.float32
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index], self.targets[index]
