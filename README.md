# English → Gujarati Neural Machine Translation

A 60.5M-parameter Transformer trained from scratch in PyTorch. FastAPI serves translations on CPU, Streamlit provides
the UI, and GitHub Actions tests and packages both.

**What is your name?** → તમારું નામ શું છે?

On the original 2,000-sentence Kaggle test split, beam search (beam 5) scored **16.68 BLEU**, **40.48 chrF++**, and
**74.96 TER**. Training used the first million corpus pairs. The frozen set in `data/eval/` overlaps the original
training data, so use the Kaggle scores when reporting this checkpoint’s quality. Translation is sentence by sentence,
and quality is limited on long, informal, or specialized text. [Model card](MODEL_CARD.md)

## Quick start

Download the 122 MB self-contained checkpoint and start the API and UI with Docker Compose. If you already have a
checkpoint, place it at `models/best.pt` and run Compose.

```bash
mkdir -p models
curl -fL https://huggingface.co/ansh164/english-gujarati-transformer/resolve/main/best.pt -o models/best.pt
docker compose up --build
```

Open the [Streamlit UI](http://localhost:8501) or [API docs](http://localhost:8000/docs).

```bash
curl -fsS http://localhost:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"What is your name?","beam_size":5}'
```

`POST /translate` returns the translation, model version, request ID, latency, and sentence count. The default limits
are 2,000 characters, 40 sentences, and beam sizes 1–8. `POST /feedback` accepts a rating and an optional correction.
Use `/health` for liveness, `/ready` for model readiness, `/info` for model and revision details, and `/metrics` for
Prometheus. [Serving configuration](configs/serve.yaml)

For a native run, use `python -m pip install -e ".[serve]"`, then run
`EN_GU_MODEL_PATH=models/best.pt python -m en_gu serve` and `python -m streamlit run app.py` in separate terminals.

## Train and test

For training, place `train.en` and `train.gu` under `en-gu/`. Run the full pipeline on a GPU, or use the smoke config for
a small CPU run. [Training dataset](https://www.kaggle.com/datasets/parvmodi/english-to-gujarati-machine-translation-dataset) ·
[Kaggle runner](notebooks/kaggle_runner.ipynb) · [DVC stages](dvc.yaml)

```bash
python -m pip install -e ".[serve,train,dev,monitor]"
python -m en_gu all
python -m en_gu all --config configs/base.yaml --config configs/smoke.yaml
python -m pytest
python -m ruff check .
python -m ruff format --check src tests
```

The pipeline cleans the corpus, trains the tokenizer and model, evaluates on the frozen split, and exports
`artifacts/best.pt`. Tests use synthetic data and tiny checkpoints. MLflow records runs and tracks the
`champion` and `previous` aliases. The promotion gate checks evaluation isolation, translation quality, checkpoint
size, and CPU latency. [Training and gate configuration](configs/base.yaml)

```bash
python -m en_gu backfill-run1 --checkpoint models/best.pt
python -m en_gu promote --candidate artifacts/best.pt --dry-run
python -m en_gu fetch-model
```

Run `backfill-run1` once to register the published checkpoint. Remove `--dry-run` to allow promotion when the gate passes.

## Monitoring

Start the Compose monitoring profile for [Grafana](http://localhost:3000) and [Prometheus](http://localhost:9090).
The commands below generate an Evidently drift report from simulated traffic and run the fixed 200-pair quality canary.

```bash
mkdir -p artifacts/monitoring
docker compose --profile monitoring up -d --build
python -m pip install -e ".[api,monitor]"
python scripts/simulate_traffic.py --profile long --count 100
docker compose logs --no-log-prefix api > artifacts/monitoring/traffic.log
python -m en_gu.drift report --logs artifacts/monitoring/traffic.log --include-simulated
python -m en_gu.quality_canary --url http://localhost:8000 --min-chrf 43.44758119227103
```

Telemetry uses aggregate statistics and a keyed input hash. Raw translation text is kept only when a user submits
a correction, with a default 90-day retention period. Feedback storage is local to each Cloud Run instance.
[Golden-set baseline](monitoring/golden-baseline.json) · [Monitoring runbook](RUNBOOKS.md#monitoring)

## CI and deployment

GitHub Actions runs lint, tests, and a container smoke test, then publishes images from `main`. With Google Cloud
configured, CD deploys to staging and then to production after approval. Production uses a tagged canary and restores
the prior revision if rollout checks fail. Cloud Run supports a Hub download, a Cloud Storage mount, or a baked
checkpoint. [CI](.github/workflows/ci.yml) · [CD](.github/workflows/cd.yml) ·
[GitHub setup](scripts/setup_github.sh) · [Google Cloud setup](scripts/setup_gcp_cicd.sh) · [Runbooks](RUNBOOKS.md)

## License

MIT
