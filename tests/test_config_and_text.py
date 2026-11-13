import pytest
import yaml

from en_gu.config import deep_merge, load_config, model_config, parse_override
from en_gu.textutils import normalize_text


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello \t world\n again  ") == "hello world again"
    assert normalize_text("") == ""


def test_deep_merge_is_recursive_and_does_not_mutate():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    merged = deep_merge(base, {"a": {"y": 3}, "c": 4})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("train.max_epochs=3", {"train": {"max_epochs": 3}}),
        ("train.peak_lr=7e-4", {"train": {"peak_lr": 7e-4}}),
        ("data.max_sentences=null", {"data": {"max_sentences": None}}),
        ("paths.eval_dir=data/eval", {"paths": {"eval_dir": "data/eval"}}),
    ],
)
def test_parse_override(text, expected):
    assert parse_override(text) == expected


def test_parse_override_rejects_garbage():
    with pytest.raises(ValueError):
        parse_override("no-equals-sign")


def test_load_config_layers_files_then_overrides(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"train": {"max_epochs": 30, "eval_every": 2000}}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"train": {"max_epochs": 2}}))
    cfg = load_config([tmp_path / "a.yaml", tmp_path / "b.yaml"], ["train.eval_every=5"])
    assert cfg.train.max_epochs == 2
    assert cfg.train.eval_every == 5


def test_base_config_matches_the_trained_model():
    cfg = load_config(["configs/base.yaml"])
    assert model_config(cfg) == {
        "vocab_size": 32000,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_len": 256,
    }
