# Roadmap

Improvements ordered by impact within each area.

## Model

1. **Out-of-distribution guard**: reject photos that are not an avocado (e.g. a small binary
   "avocado / not avocado" head, or a threshold on predictive entropy / feature distance).
   Today any image gets a ripeness answer.
2. **Ordinal-aware training**: the stages are ordered, so use an ordinal loss (CORAL/CORN) or
   label smoothing toward neighbouring stages. This targets the dominant error type (adjacent stages).
3. **Deeper fine-tuning**: unfreeze `layer3` + `layer4` with a lower learning rate (1e-4) and
   cosine schedule; add early stopping.
4. **Calibration**: measure Expected Calibration Error and reliability diagrams; apply
   temperature scaling on the validation set so "80 % confidence" really means 80 %.
5. **Real-world robustness**: stronger augmentation (random resized crop, background
   replacement, white-balance shifts) and a small test set of phone photos in kitchens/markets.
6. **Lighter backbones**: benchmark EfficientNet-B0 / ConvNeXt-Tiny / MobileNetV3 for
   accuracy vs latency; export to ONNX (or TorchScript) for faster CPU inference.
7. **Better fusion prior**: divide out the class prior when multiplying posteriors and
   down-weight near-duplicate photos to avoid overconfidence.
8. **Experiment tracking**: log runs with MLflow or Weights & Biases; version data with DVC.

## Backend

1. **Persistence**: replace the in-memory session store and the CSV feedback file with SQLite /
   PostgreSQL (sessions, images, predictions, feedback). Store uploaded images (with consent)
   so feedback can become new training data.
2. **Retraining pipeline**: a script that joins feedback with stored images, builds a
   relabeled dataset and retrains / evaluates a candidate model before promoting it.
3. **Rate limiting & auth**: per-IP rate limiting on `/analyze`; API keys if the API is exposed
   publicly.
4. **Observability**: structured JSON logs, request IDs, Prometheus metrics (latency,
   confidence distribution, low-confidence rate as a drift signal).
5. **Async LLM call**: return the prediction immediately and stream/poll the insight, so a
   slow LLM never delays the main result.
6. **Model registry**: load the model by version, expose it in `/model`, allow hot reload.

## Frontend

1. **Migrate from Create React App to Vite**: CRA is deprecated. Vite gives faster builds and
   maintained tooling.
2. **TypeScript** with types generated from the OpenAPI schema (`openapi-typescript`).
3. **Component tests** with React Testing Library; an end-to-end test with Playwright.
4. **Internationalization** (English / Spanish) with `react-i18next`.
5. **PWA / offline**: installable app; optionally on-device inference with ONNX Runtime Web.
6. **Visual probability chart** of all five stages and a photo-quality hint (blur/darkness)
   before upload.

## Engineering

1. **Release the checkpoint** as a GitHub Release asset (or Git LFS) and add a
   `scripts/download_model` helper.
2. **Pre-commit hooks** (ruff, ruff-format, prettier, eslint).
3. **CI**: extend the workflow with Docker image builds and publishing to GHCR.
4. **Dependency pinning** with a lock file (`uv` or `pip-tools`) for reproducible builds.
5. **Cloud deployment**: e.g. the API on Fly.io / Render / Cloud Run and the frontend on
   Vercel / Netlify.
