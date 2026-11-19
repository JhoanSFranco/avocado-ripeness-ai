# Architecture

## Layers

The Python package is organized so each layer depends only on the layers below it:

```text
api/          HTTP: FastAPI app factory, request/response schemas, routes
  │
services/     Use cases: AnalysisService, SessionStore, FeedbackStore
  │
insights/     Prediction -> human-readable advice (LLM or local rules)
ml/           Model, checkpoint I/O, preprocessing, predictor, fusion
data/         Dataset download/preparation, metadata & targets, PyTorch dataset
training/     Training CLI and loops (uses ml/ and data/)
  │
domain.py     Ripeness stages, confidence thresholds, stage advice (no dependencies)
config.py     Settings (environment / .env)
```

Training and serving share the same model class, transforms and stage definitions. Before,
the notebook and the API each had their own copy of these.

## Request flow: `POST /api/v1/analyze`

1. **Validation** (`api/routes/analysis.py`): content type, size limit (`MAX_UPLOAD_MB`),
   session id format (`^[A-Za-z0-9_-]{1,64}$`).
2. **Decoding** (`services/analysis.py`): Pillow decodes the image. Its decompression-bomb
   guard stays on, and unreadable files return `400`.
3. **Prediction** (`ml/predictor.py`):
   - resize to 224×224 + ImageNet normalization;
   - the ResNet-50 backbone runs **once**;
   - the dropout heads are sampled `MC_SAMPLES` times in a single batched call;
   - mean/variance of class probabilities and of the standardized regression;
   - days = `mean × target_std + target_mean` (statistics stored in the checkpoint).
4. **Session fusion** (`services/session_store.py`, `ml/fusion.py`): the prediction is
   appended to the session, and every photo in the session is fused:
   `log P(stage | photos) = Σ log P(stage | photo_i) + const`.
5. **Insight** (`insights/`): only for reliable predictions (confidence ≥ medium threshold).
   Uses Groq if `GROQ_API_KEY` is set, otherwise local rules. Any LLM error falls back to the rules.
6. **Response**: the fused prediction, the single-photo prediction, the insight and an `image_id`
   that feedback can refer to.

## Design decisions

| Decision | Why |
|---|---|
| App factory + lifespan (`create_app`) | No work at import time; tests inject a fake predictor. |
| Model loaded once at startup | ~100 MB of weights; loading per request would be far too slow. |
| API stays up if the model fails to load | `/health` reports `degraded` and `/analyze` returns `503`, instead of a crash loop. |
| Synchronous `def` endpoint for inference | FastAPI runs it in a thread pool, so the event loop is never blocked. |
| Functional dropout for MC sampling | The old approach toggled modules into train mode, which is a race condition under concurrent requests. Now the model's state never changes. |
| Backbone computed once per image | Dropout only exists in the heads, so this gives the same distribution as re-running the whole network, about `MC_SAMPLES`× faster. |
| `weights_only=True` checkpoints | Loading a checkpoint cannot execute arbitrary pickled code. |
| Bounded session store (TTL + LRU + per-session cap) | The old global dict grew without limit. |
| Same-origin API through nginx | No CORS configuration needed in production. |
| `/api/v1` prefix | Allows breaking changes later without breaking clients. |

## Frontend

```text
src/
├── api/client.js              # fetch wrapper; all API calls in one place
├── hooks/useAnalysisSession.js # session, upload, analysis, history state
├── components/                # presentational components
├── constants.js, utils/       # stage list, colors, formatting
└── styles/global.css          # design tokens + component styles
```

The hook owns the "one session = one avocado" model. **Add another photo** keeps the session,
so the next photo is fused with the previous ones. **New avocado** clears the server session and
starts a new one.
