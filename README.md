# English → Gujarati Neural Machine Translation

A Transformer encoder-decoder trained **from scratch** in PyTorch to translate English into Gujarati, with a
Streamlit app for interactive use. Training runs on Kaggle (2×T4); inference runs on CPU.

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

## Run the app locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the model

Download **`best.pt`** from the Kaggle training output and put it next to `app.py`. It is one self-contained file
(config, weights, tokenizer and test scores), about 122 MB, and is gitignored. If `best.pt` is missing, the app
tries to download it from the Hugging Face repo set in `app.py` (`HF_REPO_ID`), which only works after you have
uploaded it there (see below).

A `best.pt` from an older run that has no tokenizer inside also works if you put the matching `spm_en_gu.model`
next to it.

### 3. Start the app

```bash
streamlit run app.py
```

Open http://localhost:8501. The app runs on CPU unless CUDA is available. If you replace `best.pt` while it is
running, the new model is loaded on the next translation.

Features: beam-size control, multi-sentence input (split into sentences before translation), one-click examples,
translation history with CSV download, and model details (parameters, device, test scores) in the sidebar.

## Training pipeline

The training code is a Python package (`src/en_gu/`) driven by YAML configs, so a run can be reproduced from the
command line on any machine, not only in a notebook.

```bash
pip install -e ".[serve,train,dev]"     # package + training + test/lint tools
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

### Training on Kaggle

Open `notebooks/kaggle_runner.ipynb` on Kaggle (GPU T4 ×2, Internet on, the dataset above as input) and
**Save & Run All**. It clones this repo, installs the package, and runs `python -m en_gu all` with the paths pointed
at `/kaggle/working`. Download `/kaggle/working/best.pt` afterwards; while training, `artifacts/train/best.pt` is
refreshed whenever validation improves, so you can grab an intermediate model. To continue a previous run, set
`train.resume_checkpoint` to its `artifacts/train/last.pt` in the notebook's Settings cell.

### Tests and linting

```bash
python -m pytest        # 41 tests, ~30 s on CPU, no data or model download needed
python -m ruff check .
```

The tests cover config handling, corpus cleaning, the leak-free split and frozen eval set, batching, the model and
decoding, checkpoint loading, and the full pipeline on a small synthetic corpus. `pre-commit install` enables the
same checks (plus a guard against committing large files such as checkpoints) before each commit.

## Publish the model to Hugging Face (optional)

```bash
python upload_model.py
```

Uploads `best.pt` to the repo named in the script and asks for your token. Don't commit the token.

## Project layout

```
app.py                   Streamlit front end
src/en_gu/               Package: model.py (Transformer, decoding, loader), data.py, tokenizer.py,
                         train.py, evaluate.py, export.py, config.py, cli.py
configs/                 base.yaml (run 1 hyperparameters), smoke.yaml (tiny CPU run)
tests/                   pytest suite
notebooks/kaggle_runner.ipynb    Thin notebook that runs the pipeline on Kaggle
notebooks/archive/       Run 1 notebook with its training log
dvc.yaml                 The pipeline as DVC stages
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
