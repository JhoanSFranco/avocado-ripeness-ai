"""Convert a checkpoint produced by the original research notebook into the
current format (English keys, plain Python types, loadable with
`weights_only=True`).

Usage:
    python -m avocado_ai.ml.convert_checkpoint OLD.pth NEW.pth

Security note: the old format stores NumPy objects, so it has to be read
with full pickle support. Only convert checkpoints you trust.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from avocado_ai.ml.model import AvocadoRipenessNet, ModelMetadata, save_checkpoint

LEGACY_MODEL_CLASS = "ResNet50AguacateSoloImagen"

# Legacy metric key -> new metric key
_METRIC_KEYS = {
    "val_loss_min": "val_loss_best",
    "val_acc_final": "val_accuracy_final",
    "val_acc_max": "val_accuracy_best",
    "val_mae_final": "val_mae_final",
    "val_mae_min": "val_mae_best",
    "test_acc": "test_accuracy",
    "test_mae": "test_mae",
    "test_rmse": "test_rmse",
}

# Legacy history key -> new history key
_HISTORY_KEYS = {
    "train_loss_total": "train_loss",
    "train_loss_clf": "train_loss_classification",
    "train_loss_reg": "train_loss_regression",
    "val_loss_total": "val_loss",
    "val_loss_clf": "val_loss_classification",
    "val_loss_reg": "val_loss_regression",
    "val_acc": "val_accuracy",
    "val_mae": "val_mae",
}


def convert(source: Path, destination: Path) -> Path:
    legacy = torch.load(source, map_location="cpu", weights_only=False)
    if legacy.get("model_class") != LEGACY_MODEL_CLASS:
        raise ValueError(f"{source} is not a {LEGACY_MODEL_CLASS} checkpoint")

    metadata = ModelMetadata(
        num_classes=int(legacy["n_clases"]),
        dropout=float(legacy.get("dropout_p", 0.4)),
        target_name="days_until_overripe",
        target_mean=float(legacy["reg_mean"]),
        target_std=float(legacy["reg_std"]),
        metrics={
            new: float(legacy[old])
            for old, new in _METRIC_KEYS.items()
            if legacy.get(old) is not None
        },
        training={"epochs": legacy.get("epocas"), "seed": legacy.get("seed")},
    )
    history = {
        new: [float(x) for x in legacy.get("historial", {}).get(old, [])]
        for old, new in _HISTORY_KEYS.items()
    }

    model = AvocadoRipenessNet(num_classes=metadata.num_classes, dropout=metadata.dropout)
    model.load_state_dict(legacy["model_state_dict"])  # strict: fails on any mismatch
    return save_checkpoint(destination, model, metadata, history)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("source", type=Path, help="Legacy checkpoint (.pth)")
    parser.add_argument("destination", type=Path, help="Output checkpoint (.pth)")
    args = parser.parse_args()
    path = convert(args.source, args.destination)
    print(f"Converted checkpoint written to {path}")


if __name__ == "__main__":
    main()
