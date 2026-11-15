"""Model registry on top of MLflow: versions, aliases and checkpoint lookup.

Aliases: `champion` is the model that should be served, `previous` is the champion it replaced (kept for
rollback), `candidate` is a version waiting for a verdict from the promotion gate.
"""

import json
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

from en_gu.config import artifact_path
from en_gu.data import sha256_file
from en_gu.tracking import configure

CHAMPION, PREVIOUS, CANDIDATE = "champion", "previous", "candidate"


def _missing(exc):
    return getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST"


def _alias_missing(exc):
    # mlflow reports an alias that isn't set as INVALID_PARAMETER_VALUE rather than RESOURCE_DOES_NOT_EXIST
    return _missing(exc) or ("alias" in str(exc) and "not found" in str(exc))


def model_name(cfg):
    return cfg["tracking"]["registered_model"]


def import_run(cfg, checkpoint, *, run_name, params=None, tags=None, history=None, metrics=None, artifacts=()):
    """Create a finished run that holds a checkpoint made elsewhere (e.g. on Kaggle) with its training curve."""
    _, experiment_id = configure(cfg)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_params({k: str(v)[:500] for k, v in (params or {}).items()})
        mlflow.set_tags({k: str(v) for k, v in (tags or {}).items()})
        for point in history or []:
            mlflow.log_metrics({k: v for k, v in point.items() if k not in ("step", "epoch")}, step=point["step"])
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(checkpoint), artifact_path="model")
        for path, subdir in artifacts:
            mlflow.log_artifact(str(path), artifact_path=subdir)
    return run.info.run_id


def register_version(cfg, run_id, *, tags=None, description=None):
    """Register the checkpoint logged under `model/` in a run as a new version of the registered model."""
    client, _ = configure(cfg)
    name = model_name(cfg)
    try:
        client.get_registered_model(name)
    except MlflowException as exc:
        if not _missing(exc):
            raise
        client.create_registered_model(name)
    return client.create_model_version(
        name, source=f"runs:/{run_id}/model", run_id=run_id, tags=tags or {}, description=description
    )


def get_by_alias(cfg, alias):
    client, _ = configure(cfg)
    try:
        return client.get_model_version_by_alias(model_name(cfg), alias)
    except MlflowException as exc:
        if _alias_missing(exc):
            return None
        raise


def set_alias(cfg, alias, version):
    client, _ = configure(cfg)
    client.set_registered_model_alias(model_name(cfg), alias, str(version))


def clear_alias(cfg, alias):
    client, _ = configure(cfg)
    client.delete_registered_model_alias(model_name(cfg), alias)


def make_champion(cfg, version):
    """Serve `version`; the model it replaces stays reachable as `previous` so a rollback is one command."""
    current = get_by_alias(cfg, CHAMPION)
    if current is not None and str(current.version) != str(version):
        set_alias(cfg, PREVIOUS, current.version)
    set_alias(cfg, CHAMPION, version)


def rollback(cfg):
    """Swap `champion` and `previous`. Returns the version that is now the champion."""
    previous = get_by_alias(cfg, PREVIOUS)
    if previous is None:
        raise RuntimeError("There is no previous version to roll back to.")
    make_champion(cfg, previous.version)
    return previous


def list_versions(cfg):
    client, _ = configure(cfg)
    versions = client.search_model_versions(f"name='{model_name(cfg)}'")
    return sorted(versions, key=lambda v: int(v.version))


def find_version_by_tag(cfg, key, value):
    return next((v for v in list_versions(cfg) if v.tags.get(key) == value), None)


def scores_of(version):
    """The scores recorded for a version (see `gate.py` for the layout); empty if none."""
    return json.loads(version.tags.get("scores", "{}"))


def download_checkpoint(cfg, version):
    """Local path to a version's best.pt, downloaded once and verified against its recorded hash."""
    target = artifact_path(cfg, "registry", f"v{version.version}")
    path = target / "best.pt"
    expected = version.tags.get("model_sha256")
    if not (path.exists() and (expected is None or sha256_file(path) == expected)):
        target.mkdir(parents=True, exist_ok=True)
        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=str(target)))
        files = [downloaded] if downloaded.is_file() else sorted(f for f in downloaded.rglob("*") if f.is_file())
        if not files:
            raise FileNotFoundError(f"Version {version.version} has no checkpoint file under {version.source}")
        found = next((f for f in files if f.name == "best.pt"), files[0])
        if found != path:
            path.write_bytes(found.read_bytes())
    if expected is not None and sha256_file(path) != expected:
        raise RuntimeError(f"Checkpoint for version {version.version} does not match its recorded hash.")
    return path


def log_gate_run(cfg, report, candidate_sha256):
    """Keep the gate's verdict (and the numbers behind it) in MLflow, whether or not the candidate passed."""
    _, experiment_id = configure(cfg)
    champion = report.get("champion_overall") or {}
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"gate-{candidate_sha256[:8]}") as run:
        mlflow.set_tags(
            {
                "stage": "gate",
                "candidate_sha256": candidate_sha256,
                "gate": "passed" if report["passed"] else "rejected",
                "comparison": report["mode"],
            }
        )
        mlflow.log_metrics(
            {
                "candidate_chrf": report["candidate"]["overall"]["chrf"],
                "candidate_bleu": report["candidate"]["overall"]["bleu"],
                **{f"champion_{k}": v for k, v in champion.items() if k in ("chrf", "bleu")},
            }
        )
        mlflow.log_dict(report, "gate_report.json")
    return run.info.run_id
