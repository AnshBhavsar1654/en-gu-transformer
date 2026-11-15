# Model card: English → Gujarati Transformer (run 1, registry version 1)

## Model

| | |
|---|---|
| Task | Sentence-level machine translation, English → Gujarati |
| Architecture | Transformer encoder-decoder, 6 + 6 layers, 8 heads, d_model 512, feed-forward 2048, pre-LayerNorm |
| Parameters | 60.5M (input embedding, decoder input embedding and output projection are shared) |
| Vocabulary | Joint English+Gujarati SentencePiece unigram, 32,000 pieces, byte fallback (no unknown tokens) |
| Max length | 256 tokens |
| Decoding | Beam search, beam 5 by default |
| Format | One self-contained `best.pt` (config, fp16 weights, tokenizer, scores), 122 MB |
| Trained | On Kaggle (2×T4, ~3.1 h) with the notebook archived in `notebooks/archive/`, before the pipeline in `src/en_gu/` existed; that code is a refactor of the notebook |

## Intended use

- Understanding the gist of general-domain English text (news-like and conversational sentences) in Gujarati.
- A portfolio / learning project for building and operating an NMT system end to end.

**Not intended for:** legal, medical, financial or safety-critical text; official or published translations without
review by a Gujarati speaker; names, addresses or numbers that must be exact.

## Training data

- [English to Gujarati Machine Translation Dataset](https://www.kaggle.com/datasets/parvmodi/english-to-gujarati-machine-translation-dataset)
  (about 3.07M pairs, web-mined). Run 1 used its **first 1M pairs**; 990,453 survived cleaning and deduplication and
  976,296 were used for training.
- Cleaning drops empty or identical pairs, pairs with very different lengths, sentences over 80 words, English text
  that contains Gujarati script, and Gujarati targets that are mostly Latin script.
- The dataset's licence and the provenance of its text have not been verified. Check them before any use beyond
  personal learning.

## Evaluation

2,000 held-out sentences, scored with sacrebleu (BLEU, chrF++, TER). Test English sources were removed from training.

| Decoding | BLEU | chrF++ | TER ↓ |
|---|---|---|---|
| Greedy | 16.28 | 40.06 | 76.08 |
| Beam 5 | **16.68** | **40.48** | **74.96** |

**Important caveats**

- These numbers are on run 1's *own* test split, which cannot be regenerated: the copy of the corpus used locally
  differs slightly from the Kaggle copy, and that reshuffles the whole split. The frozen evaluation set in
  `data/eval/` is therefore a different set, and run 1 has seen many of its sentences. Scoring run 1 on its first 200
  sentences gives chrF++ 45.6 / BLEU 22.5, which is inflated by that overlap and must not be quoted as quality.
- The registry therefore keeps run 1's published scores as a *reference* only. The first model trained with the
  pipeline (which excludes the frozen eval sentences by construction) will be the first with clean scores on
  `data/eval/`, and it will also report scores by sentence length (short ≤ 10 words, medium 11–25, long ≥ 26).
- Scores by sentence length, domain or topic have not been measured for run 1.
- Only automatic metrics have been used; there has been no human evaluation.

## Speed

Measured on a developer laptop CPU with one sentence at a time (40 test sentences, beam 5): **p95 ≈ 0.8 s per
sentence**. Decoding recomputes the full decoder at every step (no key/value cache), so this is a number to
improve, not a target.

## Limitations and risks

- **Quality is modest** (BLEU ~17). Expect wrong word choices, dropped or invented content, and grammar errors,
  especially for long or unusual sentences.
- **Sentence by sentence:** the model has no context beyond a single sentence; the app splits paragraphs into
  sentences before translating.
- **Degenerate output:** like most neural MT, it can repeat words or produce fluent text unrelated to the input,
  particularly on out-of-domain input such as code, tables, or non-English text.
- **Domain bias:** the training text is web and news material, so informal speech, dialect, technical and
  domain-specific language are likely to be handled worse.
- **Social bias:** no analysis of gender, caste, religion or other biases in the data or output has been done;
  web-mined parallel text is known to contain them.
- **Entities and numbers** (names, dates, amounts) may be altered or mistranslated.

## Maintenance

- Every model is registered with its code version, data fingerprints, hyperparameters, training curve and scores
  (MLflow registry; see the README).
- A new model replaces the served one only if it passes the promotion gate (quality overall and by sentence length,
  size, latency) on the frozen evaluation set. The previous champion is kept for rollback.
