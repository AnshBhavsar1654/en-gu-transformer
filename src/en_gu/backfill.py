"""Register run 1 (trained on Kaggle before the pipeline existed) as version 1 of the model.

The training curve and test scores come from the archived notebook's printed output, and are cross-checked against
the scores embedded in best.pt, so the wrong checkpoint can't be registered by mistake.
"""

import json
import re
from pathlib import Path

from en_gu import registry
from en_gu.checkpoint import embedded_metrics
from en_gu.data import sha256_file
from en_gu.tracking import describe_run

STEP_RE = re.compile(
    r"\[step\s+(\d+) \| epoch (\d+) \| ([\d.]+)h\] val loss ([\d.]+) .*?greedy BLEU ([\d.]+) chrF\+\+ ([\d.]+)"
)
TEST_RE = re.compile(r"^\s*(greedy|beam=\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", re.M)
STOP_RE = re.compile(r"Stopped: (.+?)\. Total updates: ([\d,]+)")
SELECTED_RE = re.compile(r"Selected: (.+)")
TEST_SIZE_RE = re.compile(r"Test set \((\d+) sentences\)")


def notebook_output(path):
    """All text printed by the notebook's code cells."""
    cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    chunks = []
    for cell in cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                chunks.append("".join(out.get("text", [])))
    return "\n".join(chunks)


def parse_notebook(path):
    text = notebook_output(path)
    history = [
        {
            "step": int(m[1]),
            "epoch": int(m[2]) - 1,
            "hours": float(m[3]),
            "val_loss": float(m[4]),
            "val_bleu": float(m[5]),
            "val_chrf": float(m[6]),
        }
        for m in STEP_RE.finditer(text)
    ]
    test = {m[1]: {"bleu": float(m[2]), "chrf": float(m[3]), "ter": float(m[4])} for m in TEST_RE.finditer(text)}
    stop, selected, size = STOP_RE.search(text), SELECTED_RE.search(text), TEST_SIZE_RE.search(text)
    if not (history and "beam=5" in test and stop):
        raise ValueError(f"{path} doesn't contain the expected training log and test table.")
    return {
        "history": history,
        "test": test,
        "stop_reason": stop[1],
        "train_updates": int(stop[2].replace(",", "")),
        "selected_checkpoint": selected[1].strip() if selected else None,
        "test_sentences": int(size[1]) if size else None,
    }


def _ensure_champion(cfg, version, make_champion):
    """The first model in an empty registry becomes the champion; an existing champion is never replaced."""
    if make_champion and registry.get_by_alias(cfg, registry.CHAMPION) is None:
        registry.make_champion(cfg, version.version)
        print(f"Version {version.version} is now the champion")


def backfill(cfg, checkpoint, notebook, *, make_champion=True):
    """Create the run + registered version for run 1. Safe to repeat: an already registered file is reused."""
    checkpoint, notebook = Path(checkpoint), Path(notebook)
    parsed = parse_notebook(notebook)
    beam = parsed["test"]["beam=5"]

    embedded = embedded_metrics(checkpoint)
    for key, parsed_value in (("test_bleu", beam["bleu"]), ("test_chrf", beam["chrf"])):
        if embedded.get(key) != parsed_value:
            raise RuntimeError(
                f"{checkpoint} records {key}={embedded.get(key)} but the notebook log says {parsed_value}: "
                "these are not from the same run."
            )

    sha = sha256_file(checkpoint)
    existing = registry.find_version_by_tag(cfg, "model_sha256", sha)
    if existing is not None:
        print(f"Already registered as version {existing.version}")
        _ensure_champion(cfg, existing, make_champion)
        return existing

    params, _ = describe_run(cfg)
    scores = {
        "eval_isolated": False,
        "reference": {
            **beam,
            "greedy": parsed["test"].get("greedy"),
            "beam_size": 5,
            "sentences": parsed["test_sentences"],
            "note": "Scores on the Kaggle run's own test split. That split cannot be regenerated, so these "
            "are not directly comparable with scores on data/eval.",
        },
    }
    tags = {
        "source": "kaggle-notebook",
        "imported_from": notebook.name,
        "model_sha256": sha,
        "eval_isolated": "false",
        "selected_checkpoint": parsed["selected_checkpoint"],
        "stop_reason": parsed["stop_reason"],
        "run_label": "run1",
    }
    run_id = registry.import_run(
        cfg,
        checkpoint,
        run_name="run1-kaggle",
        params=params,
        tags=tags,
        history=parsed["history"],
        metrics={
            "train_updates": parsed["train_updates"],
            "reference_test_bleu": beam["bleu"],
            "reference_test_chrf": beam["chrf"],
            "reference_test_ter": beam["ter"],
        },
        artifacts=[(notebook, "train")],
    )
    version = registry.register_version(
        cfg,
        run_id,
        tags={**tags, "scores": json.dumps(scores)},
        description="Run 1: Transformer-base trained on the first 1M pairs on Kaggle (2xT4, ~3.1 h).",
    )
    print(f"Registered version {version.version}")
    _ensure_champion(cfg, version, make_champion)
    return version
