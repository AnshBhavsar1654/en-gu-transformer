# English → Gujarati Neural Machine Translation

A Transformer encoder-decoder trained **from scratch** in PyTorch to translate English into Gujarati, served by a
FastAPI inference service in Docker with a Streamlit front end, tested and shipped by GitHub Actions CI/CD. Training
runs on Kaggle (2×T4); inference runs on CPU.

## Results (run 1)

Evaluated on 2,000 held-out sentences. Their English sources are excluded from training.

| Decoding | BLEU | chrF++ | TER ↓ |
|---|---|---|---|
| Greedy | 16.28 | 40.06 | 76.08 |
| Beam search (beam 5) | **16.68** | **40.48** | **74.96** |

Sample translations (beam 5):

| English | Gujarati |
|---|---|
| Can I get your contact number? | શું હું તમારો સંપર્ક નંબર મેળવી શકું છું? |
| What is your name? | તમારું નામ શું છે? |
| The government has announced new schemes for farmers. | સરકારે ખેડૂતો માટે નવી યોજનાઓની જાહેરાત કરી છે. |
| India won the cricket match by five wickets. | ભારતે આ મેચ પાંચ વિકેટથી જીતી હતી. |

These are modest scores. The model was trained on about a third of the available corpus, and validation loss stopped
improving around epoch 17 (≈ 2.43), so more epochs on the same data won't help. Training on more data is the most
direct next step.

The training log and outputs of this run are kept in `notebooks/archive/run1-gujarati-english-translation.ipynb`.
See [MODEL_CARD.md](MODEL_CARD.md) for intended use, data, limitations and the caveats on these numbers.

## Model

| | |
|---|---|
| Architecture | Transformer encoder-decoder, 6 + 6 layers, 8 heads, d_model 512, feed-forward 2048 |
| Parameters | 60.5M |
| Normalisation | Pre-LayerNorm |
| Embeddings | One joint English+Gujarati SentencePiece vocabulary (32k, unigram, byte fallback), shared by the encoder input, decoder input and output projection |
| Positions | Sinusoidal, max length 256 tokens |
| Attention | Fused `scaled_dot_product_attention` |
| Decoding | Batched beam search (default beam 5) or greedy |

## Training (run 1)

| | |
|---|---|
| Data | [English to Gujarati Machine Translation Dataset](https://www.kaggle.com/datasets/parvmodi/english-to-gujarati-machine-translation-dataset): first 1M pairs (of ~3.07M); 990,453 kept after cleaning and deduplication; 976,296 used for training |
| Cleaning | Drops misaligned pairs, untranslated lines, non-Gujarati targets, over-long sentences and duplicates |
| Batching | Length-bucketed, ~24.6k tokens per update |
| Optimisation | Adam, inverse-sqrt schedule (4k warmup, peak LR 7e-4), label smoothing 0.1, gradient clipping, fp16 mixed precision |
| Run | 30 epochs, 18,450 updates, ~3.1 h on 2×T4 |
| Model selection | Last / best-validation / average of the last 5 checkpoints; the average won and was exported |

## Serving: API, UI and Docker

The model is served by a FastAPI inference service; the Streamlit app is a thin client of it.

```
Streamlit UI (app.py) ──HTTP──▶ FastAPI service (src/en_gu/api.py) ◀── models/best.pt  (registry champion)
   👍 / 👎 / corrections ──────▶   /translate /feedback /health /ready /info /metrics
```

### 1. Get the model

```bash
python -m en_gu fetch-model        # copies the registry champion to models/best.pt (+ best.pt.version.json)
```

The sidecar file carries the registry version (e.g. `v1`), which the API returns with every translation. Without a
registry, put any self-contained `best.pt` at `models/best.pt` (its version is then shown as a content hash), or set
`EN_GU_MODEL_SOURCE=hub` to download it from the Hugging Face repo, or `EN_GU_MODEL_SOURCE=registry` to load the
champion straight from MLflow.

### 2a. Run with Docker (API + UI)

```bash
docker compose up --build          # UI: http://localhost:8501   API docs: http://localhost:8000/docs
```

The UI starts once the API reports ready (model loaded). Feedback is kept in a Docker volume.

### 2b. Run without Docker

```bash
pip install -r requirements.txt    # or: pip install -e ".[serve]"
EN_GU_MODEL_PATH=models/best.pt python -m en_gu serve    # API on http://127.0.0.1:8000
streamlit run app.py                                     # UI on http://localhost:8501
STANDALONE=1 streamlit run app.py                        # UI only: loads ./best.pt (else the HF copy) in-process
```

### API

| Endpoint | Purpose |
|---|---|
| `POST /translate` | `{"text": "...", "beam_size": 5}` → `{request_id, translation, model_version, latency_ms, n_sentences}` |
| `POST /feedback` | `{"request_id", "rating": "up" \| "down", "model_version"?, "corrected_translation"?, "source_text"?, "translation"?}` |
| `GET /health` | Liveness: the process is up |
| `GET /ready` | Readiness: 503 while the model loads (or if it failed to load, with the reason), 200 with the model version after |
| `GET /info` | Model version, architecture, test scores and request limits |
| `GET /metrics` | Prometheus metrics: requests by route and status, latency histograms, model time per beam size, sentences and characters per request, feedback counts, model info and load time |

```bash
curl -X POST localhost:8000/translate -H "content-type: application/json" -d '{"text": "What is your name?"}'
# {"request_id":"…","translation":"તમારું નામ શું છે?","model_version":"v1","latency_ms":108.4,"n_sentences":1}
```

- Every response has an `X-Request-ID` header (an incoming one is reused) and `X-Model-Version` once the model is loaded.
- Input limits (in `configs/serve.yaml`): 2,000 characters, 40 sentences, beam size 1–8. Violations return 422 with a
  readable reason. Text is split into sentences (the model was trained on single sentences) and the line layout is kept.
- The model loads in a background thread, so the container answers `/health` immediately. A model that fails to load
  leaves the service up with `/ready` false and the error in its body, instead of a crash loop.
- Requests are translated one at a time per process: concurrent CPU inference only competes for the same cores.
  Throughput and concurrency will be set from load tests.
- Settings live in `configs/serve.yaml`; `EN_GU_MODEL_SOURCE`, `EN_GU_MODEL_PATH`, `EN_GU_MODEL_ALIAS`,
  `EN_GU_FEEDBACK_DB`, `EN_GU_DEVICE` and `PORT` override them.

**Feedback and privacy.** The API does not log request text. A thumbs up/down stores only the request id, rating and
model version. Text (English input, model output and the user's correction) is stored only when the user submits a
correction in the UI, which says so next to the button. Feedback rows are deleted after 90 days.

### Docker images

| Target | What | Size |
|---|---|---|
| `api` | Inference API, CPU-only torch 2.12.1, model mounted at `/models` | 994 MB |
| `api-baked` | Same with `models/best.pt` inside, for hosts without volumes (e.g. Cloud Run) | 1.12 GB |
| `ui` | Streamlit client (no torch) | 586 MB |

Both run as a non-root user. Measured on a 16-thread laptop with Docker Desktop (WSL 2), run 1 model:

| | |
|---|---|
| Cold start, new container → `/ready` | 4.4–9.6 s (18 s on the very first start, with a cold disk cache) |
| Model load inside that | ~3.5 s |
| Latency, one 10-word sentence, beam 5, 20 sequential requests | p50 207 ms, p95 307 ms |

These are local numbers, not SLOs; production targets come from load tests on the deployed service.

## Training pipeline

The training code is a Python package (`src/en_gu/`) driven by YAML configs, so a run can be reproduced from the
command line on any machine, not only in a notebook.

```bash
pip install -e ".[serve,train,tracking,dev]"     # package + training + MLflow + test/lint tools
```

| Stage | Command | What it does | Main outputs |
|---|---|---|---|
| prepare | `python -m en_gu prepare` | Clean and dedupe the corpus, split train/val/test | `data/processed/`, `data/eval/` |
| tokenizer | `python -m en_gu tokenizer` | Train the joint SentencePiece model | `artifacts/tokenizer/` |
| train | `python -m en_gu train` | Train; snapshots and `best.pt` refreshed at each validation | `artifacts/train/` |
| evaluate | `python -m en_gu evaluate` | Compare last / best / averaged checkpoints on val, score the winner on test | `artifacts/eval/metrics.json` |
| export | `python -m en_gu export` | Package the winner into one self-contained file | `artifacts/best.pt` |

`python -m en_gu all` runs every stage. Hyperparameters live in `configs/base.yaml` (they match run 1); layer another
file on top (`--config configs/base.yaml --config configs/smoke.yaml`) or override single values
(`--set train.max_epochs=3`).

- **Smoke test:** `python -m en_gu all --config configs/base.yaml --config configs/smoke.yaml` runs a tiny model on a
  20k-sentence slice on CPU in a few minutes. It writes only to `data/smoke/`. The resulting model is not useful; it
  just proves the pipeline works end to end.
- **Frozen evaluation set:** the first `prepare` writes `data/eval/` together with a hash manifest. Later runs reuse
  those files instead of regenerating them (even if the training data changes), and `evaluate` refuses to run if they
  were modified. This keeps scores comparable between models. Set `data.refreeze_eval=true` only if you accept that
  new scores can't be compared with old ones.
- **Score any checkpoint:** `python -m en_gu score best.pt --split test --beam 5` (add `--limit 200` for a quick check).
- **DVC:** `dvc.yaml` describes the same five stages for `dvc repro` (after `dvc init` and `dvc add en-gu`).

> **Caveat for run 1:** the corpus copy used to generate `data/eval/` locally differs from the Kaggle copy by at
> least one sentence pair, which changes the whole random split. The run 1 model was therefore trained on sentences
> that appear in the locally generated eval set, so its scores there would be too optimistic. Its published scores
> (above) come from the Kaggle test set. Compare future models against each other on `data/eval/`.

### Tracking, registry and promotion gate

Every pipeline run is logged to MLflow (hyperparameters, git commit, fingerprints of the training and eval data,
the training curve, test scores and `best.pt`) in a local SQLite store (`mlflow.db` and `mlartifacts/`, both
gitignored). Browse it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Tracking never stops a run: if
MLflow is missing or unreachable it logs a warning and carries on (`--set tracking.enabled=false` turns it off).

Models are kept in a registry. The alias `champion` marks the model to serve; `previous` keeps the one it replaced so
a rollback is one command. A new model becomes the champion only by passing the **promotion gate**:

```bash
python -m en_gu backfill-run1                  # once: register the Kaggle run 1 model (best.pt) as version 1
python -m en_gu promote                        # judge artifacts/best.pt against the champion
python -m en_gu promote --limit 200 --beam 1   # quick dry run on part of the test set; never changes the registry
python -m en_gu registry                       # list versions and aliases
python -m en_gu registry --rollback            # make the previous version the champion again
```

`promote` exits with a non-zero code when the candidate is rejected, so it can gate a CI job. The rules (thresholds
are in the `gate:` section of `configs/base.yaml`):

| Check | Rule |
|---|---|
| Isolation | The candidate must have been trained without the frozen eval sentences (its checkpoint records which eval set it was trained against) |
| Quality | chrF++ and BLEU on the frozen test set may not drop more than 0.5 below the champion |
| Length buckets | chrF++ in each bucket (short ≤ 10, medium 11–25, long ≥ 26 words) may not drop more than 1.5; buckets under 30 sentences are reported but not enforced |
| Size | Checkpoint at most 300 MB |
| Latency | p95 per sentence on CPU at most 1.25× the champion's, measured in the same run (optional absolute budget) |

Run 1 was trained on a split that overlaps the frozen eval set, so its scores there are inflated. While run 1 is the
champion, a candidate is compared against run 1's published Kaggle scores with a wider tolerance (chrF++ 1.5, BLEU 1.0)
and the length-bucket check is skipped. After the first model trained with this pipeline is promoted, comparisons
become like for like.

### Training on Kaggle

Open `notebooks/kaggle_runner.ipynb` on Kaggle (GPU T4 ×2, Internet on, the dataset above as input) and
**Save & Run All**. It clones this repo, installs the package, and runs `python -m en_gu all` with the paths pointed
at `/kaggle/working`. Download `/kaggle/working/best.pt` afterwards; while training, `artifacts/train/best.pt` is
refreshed whenever validation improves, so you can grab an intermediate model. To continue a previous run, set
`train.resume_checkpoint` to its `artifacts/train/last.pt` in the notebook's Settings cell.

MLflow tracking is switched off on Kaggle. After downloading `best.pt`, judge and register it locally with
`python -m en_gu promote --candidate best.pt`: the checkpoint records which eval set it was trained against, which is
what the gate's isolation check reads.

### Tests and linting

```bash
python -m pytest                 # 112 tests, a few minutes on CPU, no data or model download needed
python -m pytest -m "not smoke"  # skip the model smoke tests (they start the real API process)
python -m ruff check . && python -m ruff format --check src tests     # what CI's Lint job runs
```

The tests cover config handling, corpus cleaning, the leak-free split and frozen eval set, batching, the model and
decoding, checkpoint loading, experiment tracking, the registry, the promotion gate, the full pipeline on a small
synthetic corpus, and the inference API (validation, readiness, feedback, metrics) on a tiny random model. The
**model smoke tests** (`-m smoke`) start `python -m en_gu serve` as a separate process on a tiny random checkpoint and
check it over HTTP, exactly as CI does. `pre-commit install` enables the same checks (plus a guard against committing
large files such as checkpoints) before each commit.

## CI/CD (GitHub Actions)

```
pull request ──▶ ci.yml:  Lint ─▶ Tests and model smoke test ─▶ Docker build and container smoke test
                          (all three are required checks; main accepts changes only through pull requests)

merge to main ─▶ cd.yml:  CI again ─▶ build + push images (tag = git SHA) ─▶ staging ─▶ smoke test
                                                                     ─▶ manual approval ─▶ production ─▶ smoke test
                                                                        (failed smoke test = traffic rolled back)
```

| Workflow | What it does |
|---|---|
| `.github/workflows/ci.yml` | On every pull request: `ruff check` and `ruff format --check`; the unit tests; the model smoke tests; then builds the `api` and `ui` images, has the `api` image generate a tiny model, starts the container and runs the HTTP smoke test against it (including that `/info` reports the commit being tested). Image sizes go to the job summary. |
| `.github/workflows/cd.yml` | On every merge to `main`: re-runs CI on the merged commit, pushes `ghcr.io/anshbhavsar1654/en-gu-api` and `en-gu-ui` tagged with the git SHA (and `main`), copies the API image to Artifact Registry, then deploys to staging and, after approval, to production. |
| `.github/workflows/deploy.yml` | Reusable deploy of one image to one Cloud Run service: remembers the revision currently serving, deploys, runs the smoke test against the public URL, and shifts traffic back to the remembered revision if the test fails. |

CI never needs the 122 MB model or the corpus: `python -m en_gu make-tiny-model` builds a tiny random checkpoint in the
real format in a few seconds, so the loader, tokenizer, beam search and API are all exercised. The translations are
meaningless; the real model's quality is checked by the promotion gate, not by CI.

**Smoke test.** `en_gu/smoke.py` uses only the standard library, so it runs anywhere:

```bash
python -m en_gu smoke-test --url http://127.0.0.1:8000                    # any running API
PYTHONPATH=src python -m en_gu.smoke --url https://… --expect-build <git sha> --no-feedback
```

It checks `/health`, waits for `/ready`, compares `/info`'s `build` with the expected commit (the image gets it from
`--build-arg GIT_SHA`), sends a two-sentence translation and checks the request id and version headers, checks that
bad input gets 422, stores a thumbs-up (skipped with `--no-feedback`, as in production) and reads `/metrics`. It prints
one line per check and exits 1 if any check fails.

**One-time setup** (both scripts are idempotent):

1. `bash scripts/setup_github.sh`: branch protection on `main` (pull requests only, the three CI checks required, no
   force pushes, admins included) and the `staging` and `production` environments, with you as the required reviewer
   for production. Run it after CI has run once, so GitHub knows the check names.
2. `PROJECT_ID=<gcp-project> bash scripts/setup_gcp_cicd.sh`: Artifact Registry repository, a deployer service
   account with only `run.admin` and `artifactregistry.writer`, and Workload Identity Federation that trusts only this
   repository, so **no cloud key is stored in GitHub**. It sets the repository variables `GCP_PROJECT_ID`,
   `GCP_REGION`, `GCP_AR_REPO`, `GCP_WIF_PROVIDER` and `GCP_DEPLOY_SA`.

Until the GCP variables exist, CD stops after publishing the images to GHCR and says so in the run summary. Optional
variables: `STAGING_SERVICE` / `PRODUCTION_SERVICE` (default `en-gu-api-staging` / `en-gu-api`) and
`EN_GU_MODEL_SOURCE` (default `hub`: the service downloads `best.pt` from Hugging Face at startup). Cloud Run sizing,
canary traffic splits and the cold-start comparison with the `api-baked` image are Phase 4.

## Publish the model to Hugging Face (optional)

```bash
python upload_model.py
```

Uploads `best.pt` to the repo named in the script and asks for your token. Don't commit the token.

## Project layout

```
app.py                   Streamlit front end (client of the API, or STANDALONE=1)
src/en_gu/               Package: model.py (Transformer, decoding, loader), data.py, tokenizer.py,
                         train.py, evaluate.py, export.py, config.py, cli.py,
                         tracking.py, registry.py, gate.py, promote.py, backfill.py,
                         api.py, serving.py, feedback.py, metrics.py, textutils.py,
                         smoke.py (HTTP smoke test), tinymodel.py (tiny random model for tests and CI)
configs/                 base.yaml (run 1 hyperparameters), smoke.yaml (tiny CPU run), serve.yaml (API)
Dockerfile               Targets: api, api-baked, ui
docker-compose.yml       Local stack: API + UI
.github/workflows/       ci.yml (pull requests), cd.yml (main -> staging -> production), deploy.yml (reusable)
scripts/                 One-time setup: setup_github.sh (branch protection, environments), setup_gcp_cicd.sh
tests/                   pytest suite
notebooks/kaggle_runner.ipynb    Thin notebook that runs the pipeline on Kaggle
notebooks/archive/       Run 1 notebook with its training log
dvc.yaml                 The pipeline as DVC stages
MODEL_CARD.md            Intended use, data, evaluation caveats and limitations
upload_model.py          Push best.pt to Hugging Face
pyproject.toml  Makefile  .pre-commit-config.yaml  requirements.txt
```

`gujarati-english-translation.v1-backup.ipynb` and `transformer_en_gu.pth` are the earlier 4-layer, word-level v1 and
are not used by the app.

## Limitations

- Trained on a subset of the corpus; quality is limited (see Results).
- The corpus is web-mined, so translations lean towards news and general web text and may handle domain-specific or
  informal language poorly.
- Inputs are translated sentence by sentence, so cross-sentence context is not used.
- CPU inference is slow for long text and large beams.

## License

MIT
