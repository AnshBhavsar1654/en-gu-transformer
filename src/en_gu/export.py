"""Stage 5: package the selected weights, config, tokenizer and metrics into one `best.pt` for the app."""

import json

import torch

from en_gu.checkpoint import save_for_app
from en_gu.config import artifact_path, model_config
from en_gu.model import load_for_inference, translate
from en_gu.tokenizer import load_tokenizer, spm_model_path


def export(cfg):
    eval_dir = artifact_path(cfg, "eval")
    state = torch.load(eval_dir / "selected.pt", map_location="cpu", weights_only=True)
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    sp = load_tokenizer(spm_model_path(cfg))

    out = artifact_path(cfg, "best.pt")
    save_for_app(
        out,
        model_config(cfg),
        state,
        sp,
        {
            k: metrics[k]
            for k in (
                "checkpoint",
                "train_updates",
                "test_bleu",
                "test_chrf",
                "test_ter",
                "test_sentences",
            )
        },
    )
    print(f"Saved {out} ({out.stat().st_size / 1e6:.0f} MB)")

    # Sanity check: reload exactly the way app.py does
    model, sp2, recorded = load_for_inference(out, "cpu")
    print(recorded)
    print(translate(model, sp2, ["Can I get your contact number?"], beam_size=2))
    return out
