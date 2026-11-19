"""Training / evaluation loops for the multi-task model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from avocado_ai.ml.model import AvocadoRipenessNet


@dataclass
class MultiTaskLoss:
    """total = cross_entropy(stage) + lambda_reg * smooth_l1(standardized days)."""

    classification: nn.Module
    regression: nn.Module
    lambda_reg: float = 0.5

    def __call__(self, logits, regression, labels, targets):
        loss_clf = self.classification(logits, labels)
        loss_reg = self.regression(regression, targets)
        return loss_clf + self.lambda_reg * loss_reg, loss_clf, loss_reg


@dataclass
class EpochStats:
    loss: float
    loss_classification: float
    loss_regression: float
    accuracy: float
    mae_days: float


def run_epoch(
    model: AvocadoRipenessNet,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    target_std: float,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 1.0,
) -> EpochStats:
    """One pass over `loader`. Trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)

    totals = np.zeros(3)  # total, classification, regression (sums of batch means)
    correct = 0
    abs_error_days = 0.0
    seen = 0

    with torch.set_grad_enabled(training):
        for images, labels, targets in loader:
            images, labels, targets = images.to(device), labels.to(device), targets.to(device)
            logits, regression = model(images)
            loss, loss_clf, loss_reg = criterion(logits, regression, labels, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Clipping stabilizes the two competing task gradients.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            totals += [loss.item(), loss_clf.item(), loss_reg.item()]
            correct += (logits.argmax(dim=1) == labels).sum().item()
            # Both values are standardized; the absolute error scales by std only.
            abs_error_days += ((regression - targets).abs() * target_std).sum().item()
            seen += labels.size(0)

    means = totals / max(len(loader), 1)
    return EpochStats(
        loss=float(means[0]),
        loss_classification=float(means[1]),
        loss_regression=float(means[2]),
        accuracy=correct / max(seen, 1),
        mae_days=abs_error_days / max(seen, 1),
    )


@torch.inference_mode()
def collect_predictions(
    model: AvocadoRipenessNet, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    """Deterministic predictions (no MC sampling) for final evaluation."""
    model.eval()
    labels, preds, targets, regressions = [], [], [], []
    for images, y_clf, y_reg in loader:
        logits, regression = model(images.to(device))
        labels.append(y_clf.numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(y_reg.numpy())
        regressions.append(regression.cpu().numpy())
    return {
        "labels": np.concatenate(labels),
        "predictions": np.concatenate(preds),
        "targets": np.concatenate(targets),
        "regression": np.concatenate(regressions),
    }
