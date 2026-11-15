import json
from pathlib import Path

import pytest

from en_gu.cli import STAGES, run_stage
from en_gu.config import deep_merge
from en_gu.tracking import Tracker, configure, flatten, git_info, resolve_uri, tracked_run

mlflow = pytest.importorskip("mlflow")


def with_tracking(cfg, tmp_path):
    return type(cfg)(
        deep_merge(
            cfg,
            {
                "tracking": {
                    "enabled": True,
                    "uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                    "artifact_root": str(tmp_path / "mlartifacts"),
                    "experiment": "test",
                    "registered_model": "test-model",
                }
            },
        )
    )


def test_flatten_nests_with_dots():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_relative_sqlite_uri_becomes_absolute():
    uri = resolve_uri("sqlite:///mlflow.db")
    assert uri.startswith("sqlite:///") and Path(uri[len("sqlite:///") :]).is_absolute()
    assert resolve_uri("http://localhost:5000") == "http://localhost:5000"


def test_git_info_never_raises():
    assert isinstance(git_info(), dict)


def test_disabled_tracker_is_a_silent_no_op(tiny_cfg):
    with tracked_run(tiny_cfg, "train") as tracker:
        tracker.log_params({"a": 1})
        tracker.log_metrics({"m": 1.0}, step=1)
        tracker.set_tags({"t": "x"})
        tracker.log_artifact("does-not-exist")
        assert not tracker.enabled and tracker.run_id is None


def test_failing_store_does_not_break_the_run(tiny_cfg, tmp_path, monkeypatch):
    def broken(cfg):
        raise RuntimeError("store is down")

    monkeypatch.setattr("en_gu.tracking.configure", broken)
    tracker = Tracker(with_tracking(tiny_cfg, tmp_path), "train")
    tracker.log_metrics({"m": 1.0})  # must not raise
    assert not tracker.enabled


def test_store_in_a_new_folder_is_created(tiny_cfg, tmp_path):
    cfg = with_tracking(tiny_cfg, tmp_path)
    cfg["tracking"]["uri"] = f"sqlite:///{(tmp_path / 'new' / 'dir' / 'mlflow.db').as_posix()}"
    client, experiment_id = configure(cfg)
    assert experiment_id and (tmp_path / "new" / "dir" / "mlflow.db").exists()


@pytest.mark.slow
def test_stages_report_into_one_run(tiny_cfg, tmp_path):
    pytest.importorskip("sacrebleu")
    cfg = with_tracking(tiny_cfg, tmp_path)
    for stage in STAGES:
        run_stage(stage, cfg)

    run_id = json.loads((Path(cfg["paths"]["artifacts_dir"]) / "mlflow_run.json").read_text())["run_id"]
    client, _ = configure(cfg)
    run = client.get_run(run_id)
    assert run.data.params["model.d_model"] == "32"
    assert "paths.eval_dir" not in run.data.params  # machine-specific settings stay out
    assert len(client.get_metric_history(run_id, "val_loss")) >= 1  # one point per validation
    assert {"test_bleu", "test_chrf", "test_ter", "train_updates"} <= run.data.metrics.keys()
    assert run.data.tags["stop_reason"] and len(run.data.tags["data.eval_sha256"]) == 64
    assert run.data.tags["exported_sha256"] and run.data.tags["selected_checkpoint"]
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {"model", "eval", "train"} <= artifacts
    assert len(client.search_runs([run.info.experiment_id])) == 1  # one run, not one per stage
