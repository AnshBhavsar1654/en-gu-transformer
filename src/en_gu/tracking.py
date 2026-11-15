"""Optional MLflow experiment tracking for the pipeline.

Every call is a no-op when tracking is disabled or mlflow isn't installed, and a tracking failure only logs a
warning: it must never kill a multi-hour training run. The stages (train, evaluate, export) are separate
processes, so the run id is kept in `<artifacts_dir>/mlflow_run.json` and all three stages report into one run.
"""

import functools
import json
import logging
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from en_gu.config import artifact_path
from en_gu.data import eval_fingerprint, sha256_file

log = logging.getLogger(__name__)
RUN_FILE = "mlflow_run.json"
EXCLUDED_PARAMS = ("paths", "tracking", "gate")  # machine-specific or not part of how the model was made


def import_mlflow():
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def resolve_uri(uri):
    """Make a relative sqlite path absolute, so the store doesn't depend on the working directory."""
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        path = Path(uri[len(prefix) :])
        if not path.is_absolute():
            return prefix + path.resolve().as_posix()
    return uri


def configure(cfg):
    """Point mlflow at the configured store and return (client, experiment_id)."""
    import mlflow
    from mlflow import MlflowClient

    tcfg = cfg["tracking"]
    uri = resolve_uri(tcfg["uri"])
    if uri.startswith("sqlite:///"):  # a missing folder would make mlflow retry for minutes
        Path(uri[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(tcfg["experiment"])
    if experiment is None:
        location = Path(tcfg["artifact_root"]).resolve().as_uri()
        return client, client.create_experiment(tcfg["experiment"], artifact_location=location)
    return client, experiment.experiment_id


def flatten(mapping, prefix=""):
    flat = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def git_info():
    """Code version of this run. Empty when git (or a repository) isn't available."""

    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10).stdout.strip()

    try:
        return {"git.sha": run("rev-parse", "HEAD"), "git.dirty": str(bool(run("status", "--porcelain"))).lower()}
    except (OSError, subprocess.SubprocessError):
        return {}


def data_info(cfg):
    """Fingerprints of the data a run used, so a model can be traced back to its exact inputs."""
    info = {}
    eval_dir = Path(cfg["paths"]["eval_dir"])
    if (eval_dir / "manifest.json").exists():
        info["data.eval_sha256"] = eval_fingerprint(eval_dir)
    processed = Path(cfg["paths"]["processed_dir"])
    if (processed / "train.en").exists() and (processed / "train.gu").exists():
        info["data.train_en_sha256"] = sha256_file(processed / "train.en")
        info["data.train_gu_sha256"] = sha256_file(processed / "train.gu")
    return info


def _safe(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.enabled:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:  # tracking is best-effort
            log.warning("MLflow %s failed: %s", method.__name__, exc)
            return None

    return wrapper


class Tracker:
    def __init__(self, cfg, stage):
        self.cfg, self.stage = cfg, stage
        self.enabled, self.run_id = False, None
        if not cfg.get("tracking", {}).get("enabled", False):
            return
        self._mlflow = import_mlflow()
        if self._mlflow is None:
            log.warning("tracking.enabled is true but mlflow isn't installed (pip install -e '.[tracking]')")
            return
        try:
            self._open()
            self.enabled = True
        except Exception as exc:
            log.warning("MLflow unavailable, continuing without tracking: %s", exc)

    def _open(self):
        mlflow = self._mlflow
        _, experiment_id = configure(self.cfg)
        run_file = artifact_path(self.cfg, RUN_FILE)
        if self.stage != "train" and run_file.exists():  # evaluate / export continue the training run
            try:
                self.run_id = mlflow.start_run(run_id=json.loads(run_file.read_text())["run_id"]).info.run_id
                return
            except Exception as exc:
                log.warning("Could not resume the recorded MLflow run (%s); starting a new one", exc)
        name = f"{self.stage}-{datetime.now():%Y%m%d-%H%M}"
        self.run_id = mlflow.start_run(experiment_id=experiment_id, run_name=name).info.run_id
        run_file.parent.mkdir(parents=True, exist_ok=True)
        run_file.write_text(json.dumps({"run_id": self.run_id}), encoding="utf-8")

    @_safe
    def log_params(self, params):
        self._mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    @_safe
    def log_metrics(self, metrics, step=None):
        clean = {k.replace(" ", "_"): float(v) for k, v in metrics.items() if v is not None}
        self._mlflow.log_metrics(clean, step=step)

    @_safe
    def set_tags(self, tags):
        self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    @_safe
    def log_artifact(self, path, artifact_path=None):
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    @_safe
    def close(self):
        self._mlflow.end_run()


@contextmanager
def tracked_run(cfg, stage):
    tracker = Tracker(cfg, stage)
    try:
        yield tracker
    finally:
        tracker.close()


def describe_run(cfg):
    """Params and tags that identify a training run: hyperparameters, code version, data version."""
    params = flatten({k: v for k, v in cfg.items() if k not in EXCLUDED_PARAMS})
    return params, {**git_info(), **data_info(cfg)}
