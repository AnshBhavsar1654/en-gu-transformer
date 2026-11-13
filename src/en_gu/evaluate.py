"""Scoring (BLEU / chrF++ / TER), validation loss, and stage 4: pick the best checkpoint and test it."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from en_gu.checkpoint import atomic_save, average_state_dicts
from en_gu.config import artifact_path, model_config
from en_gu.data import load_eval_set
from en_gu.model import PAD_IDX, Transformer, load_for_inference, translate
from en_gu.tokenizer import load_tokenizer, spm_model_path


def _sacrebleu():
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError("Scoring needs sacrebleu: pip install -e '.[train]'") from exc
    return sacrebleu


@torch.no_grad()
def validation_loss(model, loader, device):
    """Token-level cross-entropy (no label smoothing)."""
    model.eval()
    total, ntokens = 0.0, 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(src, tgt_in)
        total += F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), tgt_out.view(-1), ignore_index=PAD_IDX, reduction="sum"
        ).item()
        ntokens += (tgt_out != PAD_IDX).sum().item()
    return total / ntokens


def score_translations(model, sp, sources, references, device, beam_size=1, batch_size=64):
    sacrebleu = _sacrebleu()
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        hypotheses = translate(model, sp, sources, beam_size=beam_size, batch_size=batch_size)
    return {
        "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score,
        "chrf++": sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score,
        "ter": sacrebleu.corpus_ter(hypotheses, [references]).score,
        "hypotheses": hypotheses,
    }


def _rounded(result):
    return {k: round(v, 2) for k, v in result.items() if k != "hypotheses"}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(cfg):
    """Compare last / best / averaged checkpoints on the validation set, then score the winner on test.

    Writes metrics.json, test hypotheses, and the selected weights (fp16, without the tied output
    projection) for the export stage.
    """
    device = _device()
    ecfg = cfg["evaluate"]
    train_dir = artifact_path(cfg, "train")
    (val_en, val_gu), (test_en, test_gu) = load_eval_set(cfg["paths"]["eval_dir"])
    sp = load_tokenizer(spm_model_path(cfg))

    def load_state(path):
        return torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]

    candidates = {}
    if (train_dir / "last.pt").exists():
        candidates["last"] = load_state(train_dir / "last.pt")
    if (train_dir / "best.pt").exists():
        candidates["best"] = load_state(train_dir / "best.pt")
    snapshots = sorted((train_dir / "snapshots").glob("snapshot_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[
        -cfg["train"]["num_snapshots_to_average"] :
    ]
    if len(snapshots) > 1:
        candidates[f"average of last {len(snapshots)}"] = average_state_dicts(snapshots)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {train_dir}: run the train stage first.")

    model = Transformer(**model_config(cfg)).to(device)
    val_scores = {}
    for name, state in candidates.items():
        model.load_state_dict(state)
        val_scores[name] = score_translations(model, sp, val_en, val_gu, device, beam_size=1)
        print(f"{name:>22}: BLEU {val_scores[name]['bleu']:.2f}  chrF++ {val_scores[name]['chrf++']:.2f}")

    chosen = max(val_scores, key=lambda n: val_scores[n]["chrf++"])
    model.load_state_dict(candidates[chosen])
    model.eval()
    print(f"Selected: {chosen}")

    greedy = score_translations(model, sp, test_en, test_gu, device, beam_size=1)
    beam = score_translations(model, sp, test_en, test_gu, device, beam_size=ecfg["beam_size"], batch_size=32)
    print(f"Test ({len(test_en)} sentences)   BLEU  chrF++    TER")
    for name, r in (("greedy", greedy), (f"beam={ecfg['beam_size']}", beam)):
        print(f"{name:>10}  {r['bleu']:6.2f}  {r['chrf++']:6.2f}  {r['ter']:6.2f}")

    history_path = train_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    metrics = {
        "checkpoint": chosen,
        "train_updates": history.get("step"),
        "test_sentences": len(test_en),
        "test_bleu": round(beam["bleu"], 2),
        "test_chrf": round(beam["chrf++"], 2),
        "test_ter": round(beam["ter"], 2),
        "test_greedy": _rounded(greedy),
        "test_beam": {"beam_size": ecfg["beam_size"], **_rounded(beam)},
        "validation": {name: _rounded(r) for name, r in val_scores.items()},
    }
    out = artifact_path(cfg, "eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out / "test_hypotheses.jsonl", "w", encoding="utf-8") as f:
        for src, ref, hyp in zip(test_en, test_gu, beam["hypotheses"]):
            f.write(json.dumps({"src": src, "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
    # The output projection is tied to the embedding, so it is not stored twice
    atomic_save({k: v.half() for k, v in model.state_dict().items() if k != "output_proj.weight"}, out / "selected.pt")
    return metrics


def score_checkpoint(checkpoint, cfg, split="test", beam_size=5, limit=None, spm_path=None):
    """Score any exported checkpoint (e.g. a downloaded best.pt) on the frozen val or test set."""
    device = _device()
    val, test = load_eval_set(cfg["paths"]["eval_dir"])
    sources, references = val if split == "val" else test
    if limit:
        sources, references = sources[:limit], references[:limit]
    model, sp, metrics = load_for_inference(checkpoint, device, spm_path=spm_path)
    result = score_translations(
        model, sp, sources, references, device, beam_size=beam_size, batch_size=32 if beam_size > 1 else 64
    )
    summary = {
        "checkpoint": str(Path(checkpoint)),
        "split": split,
        "sentences": len(sources),
        "beam_size": beam_size,
        **_rounded(result),
        "recorded_metrics": metrics,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
