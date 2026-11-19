"""Multi-task ResNet-50 for avocado ripeness, plus checkpoint loading and saving.

Architecture
------------
    image (B, 3, 224, 224)
      -> ResNet-50 backbone (global-average-pooled)        -> (B, 2048)
      -> shared head: FC 2048->512, ReLU, Dropout(p)
                      FC 512->128,  ReLU, Dropout(p/2)     -> (B, 128)
      -> classifier: FC 128->5   (ripeness stage logits)
      -> regressor:  FC 128->1   (standardized days until overripe; no activation
                                  because the standardized target can be negative)

Uncertainty
-----------
Monte Carlo Dropout: Dropout stays active at inference time and the heads are
sampled several times. Dropout only lives in the heads, so the (expensive)
backbone runs once per image and only the (cheap) heads are sampled. That
gives the same distribution as re-running the whole network, at a fraction of
the cost.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from avocado_ai.domain import NUM_STAGES

logger = logging.getLogger(__name__)

MODEL_CLASS_NAME = "AvocadoRipenessNet"
CHECKPOINT_FORMAT_VERSION = 2
BACKBONE_FEATURES = 2048


class AvocadoRipenessNet(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_STAGES,
        dropout: float = 0.4,
        pretrained_backbone: bool = False,
        trainable_backbone_layers: Iterable[str] | None = ("layer4",),
    ) -> None:
        """
        Args:
            num_classes: Number of ripeness stages.
            dropout: Dropout probability of the first head layer (the second uses half).
            pretrained_backbone: Start from ImageNet weights (training only; at
                inference time the checkpoint provides every weight).
            trainable_backbone_layers: Backbone blocks left trainable. Everything
                else in the backbone is frozen. `None` keeps the whole backbone trainable.
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        backbone = models.resnet50(weights=weights)

        if trainable_backbone_layers is not None:
            trainable = set(trainable_backbone_layers)
            for name, module in backbone.named_children():
                for param in module.parameters():
                    param.requires_grad = name in trainable

        # NOTE: attribute names and layer order must stay as they are so that
        # existing checkpoints keep loading (state-dict keys depend on them).
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop the FC layer
        self.shared_head = nn.Sequential(
            nn.Linear(BACKBONE_FEATURES, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
        )
        self.classifier = nn.Linear(128, num_classes)
        self.regressor = nn.Linear(128, 1)

    # ------------------------------------------------------------------ forward
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone features, shape (B, 2048)."""
        return self.features(images).flatten(start_dim=1)

    def heads(
        self, features: torch.Tensor, mc_dropout: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the shared head and both task heads.

        Dropout is applied functionally so MC sampling never mutates the
        module's train/eval state. That keeps concurrent requests thread-safe.
        """
        x = features
        for layer in self.shared_head:
            if isinstance(layer, nn.Dropout):
                x = F.dropout(x, p=layer.p, training=self.training or mc_dropout)
            else:
                x = layer(x)
        logits = self.classifier(x)
        regression = self.regressor(x).squeeze(1)
        return logits, regression

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (class logits (B, C), standardized regression (B,))."""
        return self.heads(self.extract_features(images))

    # -------------------------------------------------------------- uncertainty
    @torch.inference_mode()
    def predict_with_uncertainty(self, images: torch.Tensor, num_samples: int = 20) -> McOutput:
        """Monte Carlo Dropout prediction.

        The model must be in eval mode (BatchNorm uses running statistics).
        Returns the mean and variance of the class probabilities and of the
        standardized regression output across `num_samples` stochastic passes.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be >= 2 to estimate a variance")
        if self.training:
            raise RuntimeError("Call model.eval() before predict_with_uncertainty()")

        features = self.extract_features(images)  # (B, F): computed once
        batch_size = features.shape[0]

        # Sample all passes in one batched call: (T*B, F) -> (T, B, ...)
        logits, regression = self.heads(features.repeat(num_samples, 1), mc_dropout=True)
        probs = torch.softmax(logits, dim=-1).view(num_samples, batch_size, -1)
        regression = regression.view(num_samples, batch_size)

        return McOutput(
            probs_mean=probs.mean(dim=0),
            probs_var=probs.var(dim=0),
            regression_mean=regression.mean(dim=0),
            regression_var=regression.var(dim=0),
        )


@dataclass(frozen=True)
class McOutput:
    probs_mean: torch.Tensor  # (B, C)
    probs_var: torch.Tensor  # (B, C)
    regression_mean: torch.Tensor  # (B,) standardized
    regression_var: torch.Tensor  # (B,) standardized


# =============================================================================
# Checkpoints
# =============================================================================
@dataclass
class ModelMetadata:
    """Everything needed to rebuild and interpret a trained model."""

    num_classes: int = NUM_STAGES
    dropout: float = 0.4
    # The regression head predicts (days - target_mean) / target_std.
    target_name: str = "days_until_overripe"
    target_mean: float = 0.0
    target_std: float = 1.0
    metrics: dict[str, float | None] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    def denormalize_days(self, standardized: float) -> float:
        return standardized * self.target_std + self.target_mean


class IncompatibleCheckpointError(ValueError):
    pass


def save_checkpoint(
    path: str | Path,
    model: AvocadoRipenessNet,
    metadata: ModelMetadata,
    history: dict[str, list[float]] | None = None,
) -> Path:
    """Save a checkpoint made only of tensors and plain Python types, so it
    can be loaded with `torch.load(weights_only=True)` (no arbitrary pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_class": MODEL_CLASS_NAME,
            "state_dict": model.state_dict(),
            "metadata": _to_builtin(asdict(metadata)),
            "history": _to_builtin(history or {}),
        },
        path,
    )
    return path


def load_model(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[AvocadoRipenessNet, ModelMetadata]:
    """Load a checkpoint saved by `save_checkpoint` and return the model in eval mode."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # legacy pickles with NumPy objects end up here
        raise IncompatibleCheckpointError(
            f"Could not safely load {path}. If this is a checkpoint from the original "
            "notebook, convert it first with:\n"
            "  python -m avocado_ai.ml.convert_checkpoint <old.pth> <new.pth>"
        ) from exc

    if checkpoint.get("model_class") != MODEL_CLASS_NAME:
        raise IncompatibleCheckpointError(
            f"Expected model_class={MODEL_CLASS_NAME!r}, got {checkpoint.get('model_class')!r}"
        )

    known = ModelMetadata.__dataclass_fields__
    metadata = ModelMetadata(
        **{k: v for k, v in checkpoint.get("metadata", {}).items() if k in known}
    )
    model = AvocadoRipenessNet(
        num_classes=metadata.num_classes,
        dropout=metadata.dropout,
        pretrained_backbone=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    logger.info(
        "Loaded %s from %s (classes=%d, test_acc=%s, test_mae=%s)",
        MODEL_CLASS_NAME,
        path,
        metadata.num_classes,
        metadata.metrics.get("test_accuracy"),
        metadata.metrics.get("test_mae"),
    )
    return model, metadata


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_builtin(value: Any) -> Any:
    """Recursively convert NumPy / tensor scalars into plain Python types."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.item()
    if hasattr(value, "item") and not isinstance(value, torch.Tensor):  # NumPy scalar
        return value.item()
    return value
