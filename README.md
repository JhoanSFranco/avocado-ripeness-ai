# 🥑 Avocado Ripeness AI

Estimate the **ripeness stage** and the **remaining days until overripe** of a Hass avocado
from a photo, with **calibrated uncertainty**.

A multi-task ResNet-50 classifies the fruit into five ripening stages and, at the same time,
regresses how many days it has left. Monte Carlo Dropout turns each prediction into a
distribution, so the app can tell a confident answer from a guess and ask for another photo
when it is unsure. Several photos of the same avocado are fused with a Bayesian update.

| Metric (held-out test set, unseen fruits) | Value |
|---|---|
| Stage classification accuracy (5 classes) | **78.9 %** |
| Days-until-overripe MAE | **2.60 days** |
| Days-until-overripe RMSE | **3.35 days** |

See the [model card](docs/MODEL_CARD.md) for details and limitations.

---

## Features

- **Multi-task model**: one backbone, two heads (stage classification + days regression).
- **Uncertainty**: Monte Carlo Dropout gives confidence, class variance and a ± in days.
- **Multi-photo fusion**: photos in the same session (one session = one avocado) are combined
  as a product of probability distributions, with a warning when they strongly disagree.
- **Actionable advice**: optional LLM explanations (Groq) with an offline rule-based fallback.
- **Feedback loop**: users can mark predictions as wrong and give the correct stage.
- **Production-minded API**: typed schemas, versioned routes, upload limits, bounded session
  memory, safe checkpoint loading, health checks, Docker images.

## Architecture

```text
┌──────────────┐  /api/v1   ┌──────────────────────────── FastAPI ───────────────────────────┐
│ React (nginx)│ ─────────▶ │ routes ─▶ AnalysisService ─▶ RipenessPredictor (ResNet-50 + MC) │
└──────────────┘            │                 │                 └▶ fusion (Bayesian update)   │
                            │                 ├▶ SessionStore (TTL, LRU, bounded)            │
                            │                 ├▶ InsightGenerator (Groq LLM │ local rules)     │
                            │                 └▶ FeedbackStore (CSV)                           │
                            └──────────────────────────────────────────────────────────────────┘
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository layout

```text
avocado-ripeness-ai/
├── src/avocado_ai/          # Python package (shared by API, training and data scripts)
│   ├── config.py            # Settings from environment / .env
│   ├── domain.py            # Ripeness stages, confidence levels, stage advice
│   ├── ml/                  # Model, checkpoints, transforms, predictor, fusion
│   ├── data/                # Download, preparation, metadata/targets, PyTorch dataset
│   ├── training/            # Training CLI and train/eval loops
│   ├── insights/            # Natural-language advice (LLM + local fallback)
│   ├── services/            # Use cases: analysis, sessions, feedback
│   └── api/                 # FastAPI app factory, schemas, routes
├── tests/                   # pytest suite (no GPU or checkpoint required)
├── frontend/                # React app + nginx config
├── docker/api.Dockerfile
├── docker-compose.yml
├── docs/                    # Architecture, model card, roadmap
├── models/                  # Checkpoints (git-ignored)
└── data/                    # Datasets & feedback (git-ignored)
```

## Quick start (Docker)

Requirements: Docker Desktop. Put the checkpoint at `models/avocado_ripeness_resnet50.pth`
(see [models/README.md](models/README.md)).

```bash
cp .env.example .env          # optional: add GROQ_API_KEY for LLM explanations
docker compose up --build
```

- App: <http://localhost:3000>
- API docs (Swagger): <http://localhost:8000/docs>

## Local development

**Backend** (Python ≥ 3.10):

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[api,train,dev]"
uvicorn avocado_ai.api.app:create_app --factory --reload
```

**Frontend** (Node ≥ 18). The dev server proxies `/api` to `localhost:8000`:

```bash
cd frontend
npm ci
npm start
```

**Tests and linting**:

```bash
pytest
ruff check src tests
```

## Training

```bash
# 1. Download the dataset (~400 MB) from Mendeley Data
python -m avocado_ai.data.download

# 2. Organize the photos by stage and build metadata.csv with the regression target
python -m avocado_ai.data.prepare

# 3. Train (defaults reproduce the published checkpoint)
python -m avocado_ai.training.train --epochs 14
```

Training writes `models/avocado_ripeness_resnet50.pth` and a `.metrics.json` report
(accuracy, MAE, RMSE, confusion matrix). Run `python -m avocado_ai.training.train --help`
for every hyperparameter. A GPU is strongly recommended.

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness; `degraded` if the model is not loaded |
| `GET` | `/api/v1/model` | Stages, thresholds and checkpoint metrics |
| `POST` | `/api/v1/analyze` | Multipart `file` (+ optional `session_id`); returns prediction and insight |
| `DELETE` | `/api/v1/sessions/{session_id}` | Forget a session's photos |
| `POST` | `/api/v1/feedback` | Record whether a prediction was correct |

```bash
curl -F "file=@avocado.jpg" -F "session_id=my-avocado" http://localhost:8000/api/v1/analyze
```

<details>
<summary>Example response (abridged)</summary>

```json
{
  "session_id": "my-avocado",
  "image_id": "3f0c…",
  "prediction": {
    "stage": 3,
    "label": "Ripe",
    "dataset_name": "Ripe (first stage)",
    "confidence": 0.86,
    "confidence_level": "high",
    "reliable": true,
    "days_until_overripe": 4.1,
    "days_uncertainty": 0.7,
    "probabilities": [0.01, 0.07, 0.86, 0.05, 0.01],
    "top_candidates": [{ "stage": 3, "label": "Ripe", "probability": 0.86 }],
    "message": "High confidence: Ripe. About 4.1 days left.",
    "images_used": 1,
    "warning": null
  },
  "insight": {
    "title": "Avocado analysis: Ripe",
    "recommendation": "This is a good time to eat it.",
    "actions": ["Eat it soon.", "…"],
    "source": "local"
  }
}
```

</details>

## Configuration

All settings are environment variables (or a `.env` file). See [.env.example](.env.example).
The most relevant ones: `MODEL_PATH`, `DEVICE`, `MC_SAMPLES`, `CONFIDENCE_HIGH`,
`CONFIDENCE_MEDIUM`, `GROQ_API_KEY`.

## Dataset

[Hass Avocado Ripening Photographic Dataset](https://data.mendeley.com/datasets/3xd9n945v8/1)
(Mendeley Data): 14,710 photos of 478 avocados, photographed daily from two sides under three
storage conditions (10 °C, 20 °C, ambient) and labeled with a 1–5 ripening index. Check the
dataset page for its license and citation before redistributing it.

## Roadmap

Planned improvements are listed in [docs/ROADMAP.md](docs/ROADMAP.md).

## Acknowledgments

We sincerely thank all previous contributors for their work, ideas, and
dedication to the project. This new optimized version also builds on the
efforts and foundations of [**@Proyecto_Sl_2026I**](https://github.com/Juramirezlop/Proyecto_SI_2026I).
We are grateful to every
collaborator who has helped improve, test, document, and move the project
forward.

### Contributors

- [![JhoanSFranco](https://avatars.githubusercontent.com/u/163791421?s=64&v=4)](https://github.com/JhoanSFranco) [**JhoanSFranco**](https://github.com/JhoanSFranco)
- [![dany25-09](https://avatars.githubusercontent.com/u/140771809?s=64&v=4)](https://github.com/dany25-09) [**dany25-09**](https://github.com/dany25-09) — Carlos Daniel García Chaparro
- [![Juramirezlop](https://avatars.githubusercontent.com/u/72463915?s=64&v=4)](https://github.com/Juramirezlop) [**Juramirezlop**](https://github.com/Juramirezlop) — Juan David Ramirez Lope
- [![SebasIron-1992](https://avatars.githubusercontent.com/u/211875118?s=64&v=4)](https://github.com/SebasIron-1992) [**SebasIron-1992**](https://github.com/SebasIron-1992) — identified in the commit history as Sebastián
