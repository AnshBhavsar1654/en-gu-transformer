import json

import pytest
import torch

mlflow = pytest.importorskip("mlflow")

from en_gu import promote as promote_module  # noqa: E402
from en_gu import registry  # noqa: E402
from en_gu.checkpoint import save_for_app  # noqa: E402
from en_gu.cli import run_stage  # noqa: E402
from en_gu.config import artifact_path  # noqa: E402
from en_gu.data import eval_fingerprint, load_eval_set, sha256_file  # noqa: E402
from en_gu.model import Transformer  # noqa: E402
from en_gu.tokenizer import load_tokenizer  # noqa: E402

from .conftest import TINY_MODEL  # noqa: E402

RUN1_REFERENCE = {"bleu": 16.68, "chrf": 40.48, "ter": 74.96}


class Lab:
    """Real (tiny) checkpoints whose scores the test controls, so the gate's decisions can be checked."""

    def __init__(self, cfg, tmp_path, spm_path, monkeypatch):
        self.cfg, self.tmp_path, self.sp = cfg, tmp_path, load_tokenizer(spm_path)
        run_stage("prepare", cfg)  # writes the frozen eval set the gate scores against
        self.eval_sha256 = eval_fingerprint(cfg["paths"]["eval_dir"])
        self.cards, self.latency, self.scored = {}, {}, []
        monkeypatch.setattr(promote_module, "score_card", self._score_card)
        monkeypatch.setattr(promote_module, "latency_p95", lambda cfg, ckpt, sents, beam: self._p95(ckpt))

    def _score_card(self, cfg, checkpoint, sources, references, beam_size, eval_sha256):
        self.scored.append(len(sources))
        return self.cards[sha256_file(checkpoint)]

    def _p95(self, checkpoint):
        return self.latency.get(sha256_file(checkpoint), 100.0)

    def candidate(self, seed, chrf, bleu=16.0, *, slices=None, isolated=True, p95=None):
        torch.manual_seed(seed)
        model = Transformer(vocab_size=400, **TINY_MODEL)
        state = {k: v.half() for k, v in model.state_dict().items() if k != "output_proj.weight"}
        path = self.tmp_path / f"candidate{seed}.pt"
        metrics = {"eval_sha256": self.eval_sha256 if isolated else "some-other-eval", "test_chrf": chrf}
        save_for_app(path, {"vocab_size": 400, **TINY_MODEL}, state, self.sp, metrics)
        slices = slices or {"short": chrf, "medium": chrf, "long": chrf}
        self.cards[sha256_file(path)] = {
            "overall": {"bleu": bleu, "chrf": chrf, "ter": 75.0},
            "slices": {k: {"n": 50, "bleu": 10.0, "chrf": v, "enforced": True} for k, v in slices.items()},
        }
        if p95 is not None:
            self.latency[sha256_file(path)] = p95
        return path

    def run1_as_champion(self):
        """A champion like the real run 1: published scores only, scored on a test set we cannot regenerate."""
        path = self.tmp_path / "run1.pt"
        path.write_bytes(b"run1")
        run_id = registry.import_run(self.cfg, path, run_name="run1")
        scores = {"eval_isolated": False, "reference": RUN1_REFERENCE}
        version = registry.register_version(
            self.cfg, run_id, tags={"model_sha256": sha256_file(path), "scores": json.dumps(scores)}
        )
        registry.make_champion(self.cfg, version.version)

    def champion_version(self):
        return registry.get_by_alias(self.cfg, registry.CHAMPION).version

    def versions(self):
        return [v.version for v in registry.list_versions(self.cfg)]


@pytest.fixture
def lab(tracking_cfg, tmp_path, spm_path, monkeypatch):
    return Lab(tracking_cfg, tmp_path, spm_path, monkeypatch)


def failed_checks(report):
    return [c["name"] for c in report["checks"] if not c["passed"]]


def test_first_isolated_model_becomes_the_champion(lab):
    report = promote_module.promote(lab.cfg, lab.candidate(1, 40.0))
    assert report["passed"] and report["promoted"] and report["mode"] == "no_champion"
    version = registry.get_by_alias(lab.cfg, registry.CHAMPION)
    assert version.tags["gate"] == "passed" and version.tags["eval_isolated"] == "true"
    scores = registry.scores_of(version)
    assert scores["eval_sha256"] == lab.eval_sha256 and scores["test"]["chrf"] == 40.0
    saved = json.loads(artifact_path(lab.cfg, "gate", "report.json").read_text(encoding="utf-8"))
    assert saved["passed"] and saved["candidate"]["eval_isolated"]


def test_worse_candidate_is_rejected_and_the_registry_is_untouched(lab):
    promote_module.promote(lab.cfg, lab.candidate(1, 40.0))
    report = promote_module.promote(lab.cfg, lab.candidate(2, 35.0))
    assert not report["passed"] and not report["promoted"]
    assert failed_checks(report) == ["chrf", "slice:short", "slice:medium", "slice:long"]
    assert lab.versions() == [1] and lab.champion_version() == 1
    assert registry.get_by_alias(lab.cfg, registry.PREVIOUS) is None


def test_better_candidate_replaces_the_champion_and_the_old_one_is_kept_for_rollback(lab):
    promote_module.promote(lab.cfg, lab.candidate(1, 40.0))
    report = promote_module.promote(lab.cfg, lab.candidate(2, 42.0, bleu=18.0))
    assert report["promoted"] and report["version"] == 2 and report["mode"] == "like_for_like"
    assert lab.champion_version() == 2
    assert registry.get_by_alias(lab.cfg, registry.PREVIOUS).version == 1
    assert registry.rollback(lab.cfg).version == 1 and lab.champion_version() == 1


def test_one_bad_length_bucket_blocks_an_otherwise_better_model(lab):
    promote_module.promote(lab.cfg, lab.candidate(1, 40.0))
    lopsided = lab.candidate(2, 42.0, slices={"short": 46.0, "medium": 42.0, "long": 36.0})
    report = promote_module.promote(lab.cfg, lopsided)
    assert not report["promoted"] and failed_checks(report) == ["slice:long"]


def test_a_slower_model_is_rejected(lab):
    promote_module.promote(lab.cfg, lab.candidate(1, 40.0, p95=100.0))
    report = promote_module.promote(lab.cfg, lab.candidate(2, 41.0, p95=200.0))
    assert not report["promoted"] and failed_checks(report) == ["latency_ratio"]
    assert promote_module.promote(lab.cfg, lab.candidate(3, 41.0, p95=100.0), with_latency=False)["promoted"]


def test_candidate_trained_on_the_eval_sentences_is_rejected(lab):
    leaky = lab.candidate(1, 50.0, isolated=False)
    report = promote_module.promote(lab.cfg, leaky)
    assert not report["promoted"] and failed_checks(report) == ["eval_isolated"]
    assert lab.versions() == []
    assert promote_module.promote(lab.cfg, leaky, assume_isolated=True)["promoted"]


def test_dry_runs_and_partial_scoring_never_change_the_registry(lab):
    path = lab.candidate(1, 40.0)
    assert promote_module.promote(lab.cfg, path, dry_run=True)["passed"]
    report = promote_module.promote(lab.cfg, path, limit=20)
    assert report["passed"] and report["dry_run"] and not report["promoted"]
    assert lab.scored[-1] == 20 and lab.versions() == []


def test_reference_champion_such_as_run1_is_compared_on_its_published_scores(lab):
    lab.run1_as_champion()
    report = promote_module.promote(lab.cfg, lab.candidate(2, 39.5, bleu=16.0))  # 1 point below, inside 1.5
    assert report["mode"] == "reference" and report["promoted"] and report["version"] == 2
    assert registry.get_by_alias(lab.cfg, registry.PREVIOUS).version == 1
    assert registry.scores_of(registry.get_by_alias(lab.cfg, registry.CHAMPION))["test"]["chrf"] == 39.5


def test_candidate_far_below_a_reference_champion_is_rejected(lab):
    lab.run1_as_champion()
    report = promote_module.promote(lab.cfg, lab.candidate(2, 20.0, bleu=5.0))
    assert not report["promoted"] and lab.champion_version() == 1
    assert {"chrf", "bleu"} <= set(failed_checks(report))


def test_an_already_registered_model_is_promoted_without_a_duplicate_version(lab):
    first, second = lab.candidate(1, 40.0), lab.candidate(2, 42.0)
    promote_module.promote(lab.cfg, first)
    promote_module.promote(lab.cfg, second)
    registry.rollback(lab.cfg)  # the champion is v1 again
    report = promote_module.promote(lab.cfg, second)  # v2 comes back
    assert report["promoted"] and report["version"] == 2 and lab.versions() == [1, 2]


def test_score_card_scores_a_real_model_by_length_bucket_and_caches_it(tracking_cfg, tmp_path, spm_path, monkeypatch):
    pytest.importorskip("sacrebleu")
    run_stage("prepare", tracking_cfg)
    _, (sources, references) = load_eval_set(tracking_cfg["paths"]["eval_dir"])
    torch.manual_seed(0)
    state = {k: v.half() for k, v in Transformer(vocab_size=400, **TINY_MODEL).state_dict().items()}
    path = tmp_path / "tiny.pt"
    save_for_app(path, {"vocab_size": 400, **TINY_MODEL}, state, load_tokenizer(spm_path), {})

    card = promote_module.score_card(tracking_cfg, path, sources, references, 1, "fp")
    assert card["overall"].keys() == {"bleu", "chrf", "ter"}
    assert sum(s["n"] for s in card["slices"].values()) == len(sources)

    def boom(*args, **kwargs):
        raise AssertionError("should have come from the cache")

    monkeypatch.setattr(promote_module, "score_translations", boom)
    assert promote_module.score_card(tracking_cfg, path, sources, references, 1, "fp") == card
