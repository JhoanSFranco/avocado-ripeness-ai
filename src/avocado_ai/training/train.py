"""Train the multi-task avocado ripeness model.

Usage:
    python -m avocado_ai.training.train --data-dir data/processed \
        --output models/avocado_ripeness_resnet50.pth

The defaults reproduce the configuration of the published checkpoint.
Writes the checkpoint plus a `<output>.metrics.json` report next to it.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from avocado_ai.data.dataset import AvocadoImageDataset
from avocado_ai.data.metadata import TARGET_COLUMN, split_by_group
from avocado_ai.domain import NUM_STAGES, RipenessStage
from avocado_ai.ml.model import AvocadoRipenessNet, ModelMetadata, resolve_device, save_checkpoint
from avocado_ai.ml.transforms import eval_transform, train_transform
from avocado_ai.training.engine import MultiTaskLoss, collect_predictions, run_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the avocado ripeness model.")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output", type=Path, default=Path("models/avocado_ripeness_resnet50.pth"))
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-reg", type=float, default=0.5, help="Weight of the regression loss.")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr-step", type=int, default=5, help="StepLR step size (epochs).")
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument(
        "--trainable-layers",
        nargs="*",
        default=["layer4"],
        help="ResNet blocks to fine-tune (e.g. layer3 layer4).",
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_frequency_weights(labels: pd.Series, num_classes: int) -> torch.Tensor:
    """Class weights ∝ 1/frequency, rescaled so they sum to `num_classes`."""
    counts = labels.value_counts()
    weights = torch.tensor([1.0 / counts.get(i, 1) for i in range(num_classes)])
    return weights / weights.sum() * num_classes


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    # --- Data ------------------------------------------------------------------
    metadata = pd.read_csv(args.data_dir / "metadata.csv").dropna(subset=[TARGET_COLUMN])
    train_df, val_df, test_df = split_by_group(metadata, seed=args.seed)
    print(
        f"Split (photos): train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} | "
        f"fruits: {train_df['group_id'].nunique()}/{val_df['group_id'].nunique()}/"
        f"{test_df['group_id'].nunique()}"
    )

    # Target standardization statistics come from the training split only.
    target_mean = float(train_df[TARGET_COLUMN].mean())
    target_std = float(train_df[TARGET_COLUMN].std()) + 1e-8

    def loader(frame: pd.DataFrame, train: bool) -> DataLoader:
        dataset = AvocadoImageDataset(
            frame,
            image_root=args.data_dir,
            transform=train_transform() if train else eval_transform(),
            target_mean=target_mean,
            target_std=target_std,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=train,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    train_loader, val_loader, test_loader = (
        loader(train_df, True),
        loader(val_df, False),
        loader(test_df, False),
    )

    # --- Model & optimization ----------------------------------------------------
    model = AvocadoRipenessNet(
        num_classes=NUM_STAGES,
        dropout=args.dropout,
        pretrained_backbone=True,
        trainable_backbone_layers=args.trainable_layers,
    ).to(device)

    class_weights = inverse_frequency_weights(train_df["stage"] - 1, NUM_STAGES).to(device)
    criterion = MultiTaskLoss(
        classification=nn.CrossEntropyLoss(weight=class_weights),
        regression=nn.SmoothL1Loss(),
        lambda_reg=args.lambda_reg,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    # --- Training loop (keep the weights with the lowest validation loss) -------
    history: dict[str, list[float]] = {
        key: []
        for key in (
            "train_loss",
            "train_loss_classification",
            "train_loss_regression",
            "val_loss",
            "val_loss_classification",
            "val_loss_regression",
            "val_accuracy",
            "val_mae",
        )
    }
    best_loss, best_state = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, criterion, device, target_std, optimizer)
        val = run_epoch(model, val_loader, criterion, device, target_std)
        scheduler.step()

        for prefix, stats in (("train", train), ("val", val)):
            history[f"{prefix}_loss"].append(stats.loss)
            history[f"{prefix}_loss_classification"].append(stats.loss_classification)
            history[f"{prefix}_loss_regression"].append(stats.loss_regression)
        history["val_accuracy"].append(val.accuracy)
        history["val_mae"].append(val.mae_days)

        improved = val.loss < best_loss
        if improved:
            best_loss, best_state = val.loss, copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train.loss:.4f} | "
            f"val loss {val.loss:.4f} acc {val.accuracy:.4f} MAE {val.mae_days:.2f}d"
            + ("  <- best" if improved else "")
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Test evaluation -----------------------------------------------------------
    results = collect_predictions(model, test_loader, device)
    test_accuracy = float((results["labels"] == results["predictions"]).mean())
    errors_days = (results["regression"] - results["targets"]) * target_std
    test_mae = float(np.abs(errors_days).mean())
    test_rmse = float(np.sqrt((errors_days**2).mean()))

    names = [s.info.dataset_name for s in RipenessStage]
    print(
        f"\nTest accuracy: {test_accuracy:.4f} | "
        f"MAE: {test_mae:.2f} days | RMSE: {test_rmse:.2f} days"
    )
    print(
        classification_report(
            results["labels"],
            results["predictions"],
            labels=list(range(NUM_STAGES)),
            target_names=names,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(
        results["labels"], results["predictions"], labels=list(range(NUM_STAGES))
    )

    # --- Save ---------------------------------------------------------------------
    metrics = {
        "val_loss_best": best_loss,
        "val_accuracy_final": history["val_accuracy"][-1],
        "val_accuracy_best": max(history["val_accuracy"]),
        "val_mae_final": history["val_mae"][-1],
        "val_mae_best": min(history["val_mae"]),
        "test_accuracy": test_accuracy,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }
    hyperparameters = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metadata_out = ModelMetadata(
        num_classes=NUM_STAGES,
        dropout=args.dropout,
        target_mean=target_mean,
        target_std=target_std,
        metrics=metrics,
        training=hyperparameters,
    )
    path = save_checkpoint(args.output, model, metadata_out, history)

    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps(
            {
                **asdict(metadata_out),
                "confusion_matrix": matrix.tolist(),
                "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Checkpoint saved to {path}\nMetrics saved to {report_path}")


if __name__ == "__main__":
    main()
