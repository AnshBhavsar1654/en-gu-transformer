"""A tiny, randomly initialised model with a small tokenizer, for tests and CI smoke tests.

CI never downloads the real 122 MB checkpoint. Instead it builds one of these in a few seconds: same file format,
same code paths (loader, tokenizer, beam search, API), meaningless translations.

    python -m en_gu make-tiny-model --out models/best.pt
"""

import random
from pathlib import Path

WORDS = (
    "the a train bus water house river market school doctor teacher farmer morning evening quick slow "
    "green red blue big small old new city village road bridge garden window door table chair book pen "
    "bread milk rice fruit tree bird fish horse dog cat friend mother father brother sister child "
    "people music story letter answer question"
).split()
TINY_MODEL = dict(d_model=32, num_heads=2, num_encoder_layers=1, num_decoder_layers=1, d_ff=64, dropout=0.0, max_len=64)
TINY_VOCAB = 400


def gujarati_word(i):
    """A fake but valid-looking Gujarati word per English word (consonant + vowel sign + consonant)."""
    return chr(0x0A95 + i % 30) + chr(0x0ABE) + chr(0x0A95 + (i * 7 + 3) % 30)


def make_pairs(n, seed=0):
    """`n` synthetic English/Gujarati sentence pairs (word-for-word, so a model can actually learn them)."""
    rng = random.Random(seed)
    en, gu = [], []
    for _ in range(n):
        idx = [rng.randrange(len(WORDS)) for _ in range(rng.randint(4, 9))]
        en.append(" ".join(WORDS[i] for i in idx).capitalize() + ".")
        gu.append(" ".join(gujarati_word(i) for i in idx))
    return en, gu


def train_tiny_tokenizer(prefix, vocab_size=TINY_VOCAB, seed=1):
    """Train a small joint SentencePiece model on synthetic pairs; returns the .model path."""
    import sentencepiece as spm

    from en_gu.model import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX

    en, gu = make_pairs(2000, seed=seed)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(en + gu),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        byte_fallback=True,
        pad_id=PAD_IDX,
        unk_id=UNK_IDX,
        bos_id=BOS_IDX,
        eos_id=EOS_IDX,
        minloglevel=2,
    )
    return prefix.with_suffix(".model")


def tiny_transformer(seed=0):
    import torch

    from en_gu.model import Transformer

    torch.manual_seed(seed)
    return Transformer(vocab_size=TINY_VOCAB, **TINY_MODEL).eval()


def save_tiny_checkpoint(path, model, spm_path, metrics=None):
    """Write `model` + tokenizer as a self-contained best.pt, exactly the format the API serves."""
    from en_gu.checkpoint import save_for_app
    from en_gu.tokenizer import load_tokenizer

    state = {k: v.half() for k, v in model.state_dict().items() if k != "output_proj.weight"}
    config = {"vocab_size": model.embedding.num_embeddings, **TINY_MODEL}
    save_for_app(path, config, state, load_tokenizer(spm_path), metrics or {"test_bleu": 1.5})
    return Path(path)


def make_tiny_checkpoint(out, seed=0):
    """Build a tokenizer and a random tiny model, and save them as one checkpoint at `out`."""
    import tempfile

    out = Path(out)
    with tempfile.TemporaryDirectory() as tmp:
        spm_path = train_tiny_tokenizer(Path(tmp) / "tok")
        save_tiny_checkpoint(out, tiny_transformer(seed), spm_path, {"test_bleu": 0.0, "note": "tiny random model"})
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB): tiny random model for smoke tests, not for use")
    return out
