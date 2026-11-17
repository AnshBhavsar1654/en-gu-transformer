"""Shared fixtures: a synthetic parallel corpus, so tests never need the real data or the 122 MB model."""

import random
from pathlib import Path

import pytest
import sentencepiece as spm

from en_gu.config import deep_merge, load_config
from en_gu.model import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX, Transformer

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
WORDS = (
    "the a train bus water house river market school doctor teacher farmer morning evening quick slow "
    "green red blue big small old new city village road bridge garden window door table chair book pen "
    "bread milk rice fruit tree bird fish horse dog cat friend mother father brother sister child "
    "people music story letter answer question"
).split()
TINY_MODEL = dict(d_model=32, num_heads=2, num_encoder_layers=1, num_decoder_layers=1, d_ff=64, dropout=0.0, max_len=64)


def gujarati_word(i):
    """A fake but valid-looking Gujarati word per English word (consonant + vowel sign + consonant)."""
    return chr(0x0A95 + i % 30) + chr(0x0ABE) + chr(0x0A95 + (i * 7 + 3) % 30)


def make_pairs(n, seed=0):
    rng = random.Random(seed)
    en, gu = [], []
    for _ in range(n):
        idx = [rng.randrange(len(WORDS)) for _ in range(rng.randint(4, 9))]
        en.append(" ".join(WORDS[i] for i in idx).capitalize() + ".")
        gu.append(" ".join(gujarati_word(i) for i in idx))
    return en, gu


@pytest.fixture
def corpus_dir(tmp_path):
    en, gu = make_pairs(4000)
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "train.en").write_text("\n".join(en) + "\n", encoding="utf-8")
    (folder / "train.gu").write_text("\n".join(gu) + "\n", encoding="utf-8")
    return folder


@pytest.fixture
def tiny_cfg(tmp_path, corpus_dir):
    """Base config shrunk to run the whole pipeline on CPU in seconds, writing only under tmp_path."""
    overrides = [
        f"paths.corpus_dir={corpus_dir.as_posix()}",
        f"paths.processed_dir={(tmp_path / 'processed').as_posix()}",
        f"paths.eval_dir={(tmp_path / 'eval').as_posix()}",
        f"paths.artifacts_dir={(tmp_path / 'artifacts').as_posix()}",
        "data.max_sentences=null",
        "data.val_size=100",
        "data.test_size=100",
        "tokenizer.vocab_size=400",
        "tokenizer.input_sentence_size=100000",
        *(f"model.{k}={v}" for k, v in TINY_MODEL.items()),
        "train.max_tokens_per_gpu=1024",
        "train.tokens_per_update=1024",
        "train.warmup_steps=5",
        "train.max_epochs=1",
        "train.eval_every=10",
        "train.quick_val_size=20",
        "train.num_workers=0",
        "tracking.enabled=false",  # tests that exercise tracking turn it on with a temporary store
    ]
    return load_config([CONFIG_DIR / "base.yaml"], overrides)


@pytest.fixture
def spm_path(tmp_path):
    """A small SentencePiece model trained on the synthetic corpus (vocab 400, byte fallback)."""
    en, gu = make_pairs(2000, seed=1)
    prefix = tmp_path / "tok"
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(en + gu),
        model_prefix=str(prefix),
        vocab_size=400,
        model_type="unigram",
        byte_fallback=True,
        pad_id=PAD_IDX,
        unk_id=UNK_IDX,
        bos_id=BOS_IDX,
        eos_id=EOS_IDX,
        minloglevel=2,
    )
    return prefix.with_suffix(".model")


@pytest.fixture
def tiny_model():
    import torch

    torch.manual_seed(0)
    return Transformer(vocab_size=400, **TINY_MODEL).eval()


@pytest.fixture
def tracking_cfg(tiny_cfg, tmp_path):
    """tiny_cfg with MLflow switched on against a throw-away store under tmp_path."""
    tracking = {
        "enabled": True,
        "uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "artifact_root": str(tmp_path / "mlartifacts"),
        "experiment": "test",
        "registered_model": "test-model",
    }
    return type(tiny_cfg)(deep_merge(tiny_cfg, {"tracking": tracking}))


@pytest.fixture
def app_checkpoint(tmp_path, tiny_model, spm_path):
    """A self-contained best.pt (tiny random model + synthetic tokenizer), as the API would serve it."""
    from en_gu.checkpoint import save_for_app
    from en_gu.tokenizer import load_tokenizer

    path = tmp_path / "serve" / "best.pt"
    state = {k: v.half() for k, v in tiny_model.state_dict().items() if k != "output_proj.weight"}
    save_for_app(path, {"vocab_size": 400, **TINY_MODEL}, state, load_tokenizer(spm_path), {"test_bleu": 1.5})
    return path


@pytest.fixture
def serve_cfg(tiny_cfg, tmp_path, app_checkpoint):
    """tiny_cfg plus the serve section from configs/serve.yaml, pointed at the tiny checkpoint."""
    serve = load_config([CONFIG_DIR / "serve.yaml"])["serve"]
    serve["model"]["path"] = str(app_checkpoint)
    serve["feedback_db"] = str(tmp_path / "feedback.db")
    return type(tiny_cfg)(deep_merge(tiny_cfg, {"serve": serve}))
