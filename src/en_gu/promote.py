"""Score a candidate checkpoint, run the promotion gate against the champion, and promote it if it passes.

python -m en_gu promote                       # artifacts/best.pt vs the registry's champion
python -m en_gu promote --limit 200 --beam 1  # quick dry run on part of the test set (never promotes)
"""

import json
from pathlib import Path

import torch

from en_gu import registry
from en_gu.checkpoint import embedded_metrics
from en_gu.config import artifact_path
from en_gu.data import eval_fingerprint, load_eval_set, sha256_file
from en_gu.evaluate import score_translations
from en_gu.files import atomic_write
from en_gu.gate import LIKE_FOR_LIKE, REFERENCE, decide, format_report, measure_latency, slice_metrics
from en_gu.model import load_for_inference, translate


def _cache_path(cfg):
    return artifact_path(cfg, "gate", "score_cache.json")


def _load_cache(cfg):
    path = _cache_path(cfg)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_cache(cfg, cache):
    path = _cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def score_card(cfg, checkpoint, sources, references, beam_size, eval_sha256):
    """Overall and per-length-bucket scores of a checkpoint on the given sentences (cached by file + eval set)."""
    cache = _load_cache(cfg)
    key = f"{sha256_file(checkpoint)}|{eval_sha256}|{beam_size}|{len(sources)}"
    if key in cache:
        return cache[key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sp, _ = load_for_inference(checkpoint, device)
    result = score_translations(
        model, sp, sources, references, device, beam_size=beam_size, batch_size=32 if beam_size > 1 else 64
    )
    gcfg = cfg["gate"]
    card = {
        "overall": {
            "bleu": round(result["bleu"], 2),
            "chrf": round(result["chrf++"], 2),
            "ter": round(result["ter"], 2),
        },
        "slices": slice_metrics(sources, references, result["hypotheses"], gcfg["slices"], gcfg["min_slice_sentences"]),
    }
    cache[key] = card
    _save_cache(cfg, cache)
    return card


def latency_p95(cfg, checkpoint, sentences, beam_size):
    """p95 ms per sentence on the CPU (the serving target), one sentence at a time."""
    model, sp, _ = load_for_inference(checkpoint, "cpu")
    return measure_latency(model, sp, sentences[: cfg["gate"]["latency_sentences"]], beam_size, translate)["p95_ms"]


def champion_card(cfg, version, sources, references, beam_size, eval_sha256, with_latency):
    scores = registry.scores_of(version)
    path = None
    if scores.get("eval_isolated") and scores.get("eval_sha256") == eval_sha256:
        path = registry.download_checkpoint(cfg, version)
        card = {**score_card(cfg, path, sources, references, beam_size, eval_sha256), "mode": LIKE_FOR_LIKE}
    else:
        # Scored elsewhere, or on a set the model saw during training: use the scores it was published with
        card = {"overall": scores["reference"], "slices": {}, "mode": REFERENCE}
    card["version"] = int(version.version)
    card["latency_p95_ms"] = None
    if with_latency:
        path = path or registry.download_checkpoint(cfg, version)
        card["latency_p95_ms"] = latency_p95(cfg, path, sources, beam_size)
    return card


def _training_run(cfg, sha):
    """The MLflow run that exported this exact file, if it came out of this working copy's last pipeline run."""
    run_file = artifact_path(cfg, "mlflow_run.json")
    if not run_file.exists():
        return None
    try:
        client, _ = registry.configure(cfg)
        run_id = json.loads(run_file.read_text())["run_id"]
        return run_id if client.get_run(run_id).data.tags.get("exported_sha256") == sha else None
    except Exception:
        return None


def promote(
    cfg,
    candidate=None,
    *,
    beam_size=None,
    limit=None,
    dry_run=False,
    with_latency=True,
    assume_isolated=False,
    stage_candidate=False,
):
    candidate = Path(candidate) if candidate else artifact_path(cfg, "best.pt")
    beam_size = beam_size or cfg["evaluate"]["beam_size"]
    dry_run = dry_run or limit is not None  # scoring part of the test set says nothing about the whole
    eval_sha256 = eval_fingerprint(cfg["paths"]["eval_dir"])
    _, (sources, references) = load_eval_set(cfg["paths"]["eval_dir"])
    if limit:
        sources, references = sources[:limit], references[:limit]

    sha = sha256_file(candidate)
    embedded = embedded_metrics(candidate)
    isolated = assume_isolated or embedded.get("eval_sha256") == eval_sha256
    card = score_card(cfg, candidate, sources, references, beam_size, eval_sha256)
    candidate_card = {
        **card,
        "size_mb": candidate.stat().st_size / 1e6,
        "latency_p95_ms": latency_p95(cfg, candidate, sources, beam_size) if with_latency else None,
        "eval_isolated": isolated,
        "sentences": len(sources),
    }

    champion_version = registry.get_by_alias(cfg, registry.CHAMPION)
    champion = (
        champion_card(cfg, champion_version, sources, references, beam_size, eval_sha256, with_latency)
        if champion_version is not None
        else None
    )
    verdict = decide(candidate_card, champion, cfg["gate"])
    report = {
        **verdict,
        "candidate": candidate_card,
        "candidate_sha256": sha,
        "champion": champion,
        "champion_overall": champion["overall"] if champion else None,
        "beam_size": beam_size,
        "eval_sha256": eval_sha256,
        "dry_run": dry_run,
        "promoted": False,
        "version": None,
    }
    print(format_report(report))
    report_path = artifact_path(cfg, "gate", "report.json")
    atomic_write(report_path, json.dumps(report, indent=2).encode())

    if dry_run:
        print(
            "Dry run: the registry was not changed." + (" (--limit scores only part of the test set.)" if limit else "")
        )
        return report

    gate_run_id = registry.log_gate_run(cfg, report, sha)
    if not verdict["passed"]:
        print("Rejected: the champion stays.")
        return report

    scores = {
        "eval_isolated": isolated,
        "eval_sha256": eval_sha256,
        "beam_size": beam_size,
        "sentences": len(sources),
        "test": card["overall"],
        "slices": card["slices"],
        "size_mb": round(candidate_card["size_mb"], 1),
    }
    existing = registry.find_version_by_tag(cfg, "model_sha256", sha)
    if existing is not None:
        version = existing
    else:
        tags = {
            "model_sha256": sha,
            "eval_isolated": str(isolated).lower(),
            "eval_sha256": eval_sha256,
            "scores": json.dumps(scores),
            "gate": "passed",
            "gate_run_id": gate_run_id,
        }
        run_id = _training_run(cfg, sha) or registry.import_run(
            cfg, candidate, run_name=f"import-{sha[:8]}", tags={"source": "external", **tags}
        )
        version = registry.register_version(cfg, run_id, tags=tags)
    report["version"] = int(version.version)
    if stage_candidate:
        registry.set_alias(cfg, registry.CANDIDATE, version.version)
        print(f"Staged version {version.version} for shadow and canary checks.")
    else:
        registry.make_champion(cfg, version.version)
        print(f"Promoted: version {version.version} is now the champion.")
        report["promoted"] = True
    atomic_write(report_path, json.dumps(report, indent=2).encode())
    return report
