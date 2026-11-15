from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from en_gu import registry  # noqa: E402
from en_gu.backfill import backfill, parse_notebook  # noqa: E402
from en_gu.checkpoint import save_for_app  # noqa: E402
from en_gu.data import sha256_file  # noqa: E402
from en_gu.tokenizer import load_tokenizer  # noqa: E402

from .conftest import TINY_MODEL  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks/archive/run1-gujarati-english-translation.ipynb"


def fake_checkpoint(tmp_path, name="best.pt", payload=b"weights"):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def register(cfg, tmp_path, payload, tags=None):
    path = fake_checkpoint(tmp_path, f"{payload.decode()}.pt", payload)
    run_id = registry.import_run(cfg, path, run_name=payload.decode(), params={"a": 1}, tags=tags)
    return registry.register_version(cfg, run_id, tags={"model_sha256": sha256_file(path), **(tags or {})})


def test_parse_the_archived_run1_notebook():
    parsed = parse_notebook(NOTEBOOK)
    assert len(parsed["history"]) == 9
    assert parsed["history"][0] == {
        "step": 2000, "epoch": 3, "hours": 0.34, "val_loss": 3.6941, "val_bleu": 6.37, "val_chrf": 26.43,
    }  # fmt: skip
    assert parsed["test"]["beam=5"] == {"bleu": 16.68, "chrf": 40.48, "ter": 74.96}
    assert parsed["test"]["greedy"] == {"bleu": 16.28, "chrf": 40.06, "ter": 76.08}
    assert parsed["train_updates"] == 18450 and parsed["stop_reason"] == "max epochs reached"
    assert parsed["selected_checkpoint"] == "average of last 5" and parsed["test_sentences"] == 2000


def test_aliases_champion_previous_and_rollback(tracking_cfg, tmp_path):
    v1, v2 = register(tracking_cfg, tmp_path, b"one"), register(tracking_cfg, tmp_path, b"two")
    assert registry.get_by_alias(tracking_cfg, registry.CHAMPION) is None

    registry.make_champion(tracking_cfg, v1.version)
    assert registry.get_by_alias(tracking_cfg, registry.PREVIOUS) is None
    registry.make_champion(tracking_cfg, v2.version)
    assert registry.get_by_alias(tracking_cfg, registry.CHAMPION).version == v2.version
    assert registry.get_by_alias(tracking_cfg, registry.PREVIOUS).version == v1.version

    assert registry.rollback(tracking_cfg).version == v1.version
    assert registry.get_by_alias(tracking_cfg, registry.CHAMPION).version == v1.version
    assert registry.get_by_alias(tracking_cfg, registry.PREVIOUS).version == v2.version  # rollback is reversible


def test_rollback_needs_a_previous_version(tracking_cfg, tmp_path):
    registry.make_champion(tracking_cfg, register(tracking_cfg, tmp_path, b"only").version)
    with pytest.raises(RuntimeError, match="previous"):
        registry.rollback(tracking_cfg)


def test_checkpoint_round_trips_through_the_registry(tracking_cfg, tmp_path):
    version = register(tracking_cfg, tmp_path, b"payload")
    path = registry.download_checkpoint(tracking_cfg, version)
    assert path.read_bytes() == b"payload"
    path.write_bytes(b"corrupted")  # a damaged local copy is detected and replaced
    assert registry.download_checkpoint(tracking_cfg, version).read_bytes() == b"payload"


def test_find_version_by_tag_and_scores(tracking_cfg, tmp_path):
    version = register(tracking_cfg, tmp_path, b"one", tags={"scores": '{"test": {"chrf": 41.0}}'})
    found = registry.find_version_by_tag(tracking_cfg, "model_sha256", version.tags["model_sha256"])
    assert found.version == version.version and registry.scores_of(found) == {"test": {"chrf": 41.0}}
    assert registry.find_version_by_tag(tracking_cfg, "model_sha256", "nope") is None


def run1_checkpoint(tmp_path, spm_path, tiny_model, metrics):
    path = tmp_path / "run1.pt"
    state = {k: v.half() for k, v in tiny_model.state_dict().items() if k != "output_proj.weight"}
    save_for_app(path, {"vocab_size": 400, **TINY_MODEL}, state, load_tokenizer(spm_path), metrics)
    return path


def test_backfill_registers_run1_once_and_makes_it_champion(tracking_cfg, tmp_path, spm_path, tiny_model):
    path = run1_checkpoint(tmp_path, spm_path, tiny_model, {"test_bleu": 16.68, "test_chrf": 40.48})
    version = backfill(tracking_cfg, path, NOTEBOOK)
    assert version.tags["eval_isolated"] == "false" and version.tags["source"] == "kaggle-notebook"
    assert registry.scores_of(version)["reference"]["chrf"] == 40.48
    assert registry.get_by_alias(tracking_cfg, registry.CHAMPION).version == version.version

    again = backfill(tracking_cfg, path, NOTEBOOK)  # repeating it must not create a second version
    assert again.version == version.version and len(registry.list_versions(tracking_cfg)) == 1


def test_backfill_refuses_a_checkpoint_from_another_run(tracking_cfg, tmp_path, spm_path, tiny_model):
    path = run1_checkpoint(tmp_path, spm_path, tiny_model, {"test_bleu": 20.0, "test_chrf": 50.0})
    with pytest.raises(RuntimeError, match="not from the same run"):
        backfill(tracking_cfg, path, NOTEBOOK)
    assert registry.list_versions(tracking_cfg) == []
