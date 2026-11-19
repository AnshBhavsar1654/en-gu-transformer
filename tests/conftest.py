"""Shared fixtures: a synthetic parallel corpus, so tests never need the real data or the 122 MB model."""

from pathlib import Path

import pytest

from en_gu.config import deep_merge, load_config
from en_gu.tinymodel import TINY_MODEL, make_pairs, save_tiny_checkpoint, tiny_transformer, train_tiny_tokenizer

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


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
    return train_tiny_tokenizer(tmp_path / "tok")


@pytest.fixture
def tiny_model():
    return tiny_transformer()


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
    return save_tiny_checkpoint(tmp_path / "serve" / "best.pt", tiny_model, spm_path)


@pytest.fixture
def serve_cfg(tiny_cfg, tmp_path, app_checkpoint):
    """tiny_cfg plus the serve section from configs/serve.yaml, pointed at the tiny checkpoint."""
    serve = load_config([CONFIG_DIR / "serve.yaml"])["serve"]
    serve["model"]["path"] = str(app_checkpoint)
    serve["feedback_db"] = str(tmp_path / "feedback.db")
    return type(tiny_cfg)(deep_merge(tiny_cfg, {"serve": serve}))
