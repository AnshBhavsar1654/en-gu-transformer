"""The whole pipeline (prepare -> tokenizer -> train -> evaluate -> export) on a synthetic corpus."""

import json
from pathlib import Path

import pytest

pytest.importorskip("sacrebleu")

from en_gu.cli import STAGES, main, run_stage  # noqa: E402
from en_gu.config import artifact_path  # noqa: E402
from en_gu.data import sha256_file  # noqa: E402
from en_gu.model import load_for_inference, translate  # noqa: E402

pytestmark = pytest.mark.slow


def test_pipeline_end_to_end(tiny_cfg, tmp_path):
    for stage in STAGES:
        run_stage(stage, tiny_cfg)

    artifacts = Path(tiny_cfg["paths"]["artifacts_dir"])
    for name in (
        "tokenizer/spm_en_gu.model",
        "train/last.pt",
        "train/best.pt",
        "train/history.json",
        "eval/metrics.json",
        "eval/test_hypotheses.jsonl",
        "best.pt",
    ):
        assert (artifacts / name).exists(), name

    metrics = json.loads((artifacts / "eval/metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_sentences"] == 100 and metrics["train_updates"] > 0
    assert {"test_bleu", "test_chrf", "test_ter", "test_greedy", "test_beam", "validation"} <= metrics.keys()

    model, sp, recorded = load_for_inference(artifacts / "best.pt")
    assert recorded["test_chrf"] == metrics["test_chrf"]
    out = translate(model, sp, ["The train. Water."], beam_size=2)
    assert isinstance(out[0], str)

    # a rerun is deterministic where it matters: the frozen eval set does not change
    eval_dir = Path(tiny_cfg["paths"]["eval_dir"])
    before = {p.name: sha256_file(p) for p in eval_dir.iterdir()}
    run_stage("prepare", tiny_cfg)
    assert {p.name: sha256_file(p) for p in eval_dir.iterdir()} == before


def test_eval_set_stays_frozen_when_the_corpus_changes(tiny_cfg, corpus_dir):
    run_stage("prepare", tiny_cfg)
    eval_dir = Path(tiny_cfg["paths"]["eval_dir"])
    before = (eval_dir / "test.en").read_text(encoding="utf-8")

    en = (corpus_dir / "train.en").read_text(encoding="utf-8").splitlines()
    gu = (corpus_dir / "train.gu").read_text(encoding="utf-8").splitlines()
    (corpus_dir / "train.en").write_text("\n".join(en[::-1]) + "\n", encoding="utf-8")  # same data, new order
    (corpus_dir / "train.gu").write_text("\n".join(gu[::-1]) + "\n", encoding="utf-8")
    run_stage("prepare", tiny_cfg)

    assert (eval_dir / "test.en").read_text(encoding="utf-8") == before
    heldout = set(before.lower().splitlines())
    train = (Path(tiny_cfg["paths"]["processed_dir"]) / "train.en").read_text(encoding="utf-8").lower().splitlines()
    assert not heldout & set(train)


def test_score_checkpoint_on_the_frozen_eval_set(tiny_cfg):
    for stage in STAGES:
        run_stage(stage, tiny_cfg)
    from en_gu.evaluate import score_checkpoint

    summary = score_checkpoint(artifact_path(tiny_cfg, "best.pt"), tiny_cfg, split="test", beam_size=1, limit=20)
    assert summary["sentences"] == 20 and "chrf++" in summary


def test_cli_parses_overrides_and_runs_a_stage(tiny_cfg, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("en_gu.cli.run_stage", lambda name, cfg: seen.update(stage=name, epochs=cfg.train.max_epochs))
    cfg_file = tmp_path / "x.yaml"
    cfg_file.write_text("train:\n  max_epochs: 7\n", encoding="utf-8")
    base = str(Path(__file__).resolve().parent.parent / "configs" / "base.yaml")
    assert main(["train", "--config", base, "--config", str(cfg_file), "--set", "train.max_epochs=3"]) == 0
    assert seen == {"stage": "train", "epochs": 3}
