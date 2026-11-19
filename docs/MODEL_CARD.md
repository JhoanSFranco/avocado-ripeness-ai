# Model card: AvocadoRipenessNet

## Overview

| | |
|---|---|
| Task | Multi-task: ripeness stage classification (5 classes) + days-until-overripe regression |
| Input | One RGB photo of one Hass avocado, resized to 224×224 |
| Backbone | ResNet-50, ImageNet (V2) initialization; only `layer4` fine-tuned |
| Heads | Shared FC 2048→512→128 (ReLU, Dropout 0.4 / 0.2) → classifier 128→5, regressor 128→1 |
| Uncertainty | Monte Carlo Dropout (default 8 samples at serving time) |
| Checkpoint | `models/avocado_ripeness_resnet50.pth` (~99 MB) |

## Data

Hass Avocado Ripening Photographic Dataset (Mendeley Data, `3xd9n945v8`): 478 avocados
photographed daily from two sides under three storage conditions (T10 = 10 °C, T20 = 20 °C,
Tam = ambient). Labels are the 1–5 ripening index:

| Stage | Dataset name | App label |
|---|---|---|
| 1 | Underripe | Unripe |
| 2 | Breaking | Breaking |
| 3 | Ripe (first stage) | Ripe |
| 4 | Ripe (second stage) | Very ripe |
| 5 | Overripe | Overripe |

### Regression target

For each avocado, the first day it reached stage 5 is found. Each photo's target is that day minus
the photo's day, clipped at −2 days. The model learns the standardized target
`(days − mean) / std`, using the training-split statistics (mean = 6.23, std = 6.43).

Two alternatives (days until stage 3 and until stage 4) were rejected: most of their values
collapse to 0, so the regressor learns to always predict 0.

### Splits

Splits are made **by fruit**, not by photo (`GroupShuffleSplit`): roughly 70 / 15 / 15 % for
train / validation / test. Photos of the same avocado never appear in two splits, so test
results measure generalization to unseen fruits.

## Training

| Setting | Value |
|---|---|
| Loss | CrossEntropy (inverse-frequency class weights) + 0.5 × SmoothL1 |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | StepLR (step 5, γ 0.5), 14 epochs |
| Batch size | 16 |
| Augmentation | Horizontal flip, ±10° rotation, light color jitter |
| Model selection | Lowest validation loss |
| Seed | 42 |

## Results

| Split | Accuracy | MAE (days) | RMSE (days) |
|---|---|---|---|
| Validation (best epoch) | 76.4 % | 2.44 | — |
| Test (unseen fruits) | **78.9 %** | **2.60** | **3.35** |

Observations from the research notebook:

- Most errors are between **adjacent** stages (e.g. 3 ↔ 4), which are visually and biologically
  close. Extreme stages are well separated.
- Correct predictions concentrate at high confidence. Errors concentrate at medium/low
  confidence, which is why the app asks for another photo below the medium threshold.
- Higher regression uncertainty tends to go with larger absolute error.

## Intended use and limitations

- **Intended use**: consumer guidance and education. It does not replace inspecting the fruit.
- **Domain shift**: the training photos come from a controlled setup (consistent background
  and lighting). Kitchen or market photos with clutter, several fruits or unusual lighting may
  reduce accuracy. Out-of-distribution inputs (not an avocado) are not detected.
- **Variety**: trained only on Hass avocados.
- **Storage conditions**: the days estimate reflects the dataset's storage conditions. Real
  shelf life depends on temperature and handling.
- **Fusion assumption**: multi-photo fusion assumes the photos are independent pieces of evidence
  and that all stages are equally likely beforehand. Many near-duplicate photos therefore inflate
  confidence.
