"""Stage 3: train the Transformer. A direct port of the notebook training loop."""

import json
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from en_gu.checkpoint import atomic_save, save_for_app, spm_tensor
from en_gu.config import artifact_path, model_config
from en_gu.data import PairDataset, TokenBatchSampler, collate, load_eval_set, read_lines
from en_gu.evaluate import score_translations, validation_loss
from en_gu.model import PAD_IDX, Transformer
from en_gu.tokenizer import check_tokenizer, load_tokenizer, spm_model_path

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars are optional

    def tqdm(iterable, **_):
        return iterable


class LossCompute(nn.Module):
    """Computes the loss on each GPU, so DataParallel never gathers the huge logits tensor."""

    def __init__(self, model, label_smoothing):
        super().__init__()
        self.model = model
        self.label_smoothing = label_smoothing

    def forward(self, src, tgt_in, tgt_out):
        with torch.autocast("cuda", dtype=torch.float16, enabled=src.is_cuda):
            logits = self.model(src, tgt_in)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            tgt_out.view(-1),
            ignore_index=PAD_IDX,
            label_smoothing=self.label_smoothing,
            reduction="sum",
        )
        ntokens = (tgt_out != PAD_IDX).sum()
        return loss.unsqueeze(0), ntokens.unsqueeze(0)


def train(cfg):
    tcfg = cfg["train"]
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} ({num_gpus} GPU(s))")

    max_tokens = tcfg["max_tokens_per_gpu"] * max(num_gpus, 1)
    update_freq = max(1, round(tcfg["tokens_per_update"] / max_tokens))
    print(
        f"Micro-batch: {max_tokens} tokens, gradient accumulation: {update_freq} "
        f"-> ~{max_tokens * update_freq} tokens/update"
    )

    out_dir = artifact_path(cfg, "train")
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config = model_config(cfg)

    # ---- data ----
    sp = load_tokenizer(spm_model_path(cfg))
    check_tokenizer(sp, config["vocab_size"])
    processed = Path(cfg["paths"]["processed_dir"])
    train_en, train_gu = read_lines(processed / "train.en"), read_lines(processed / "train.gu")
    (val_en, val_gu), _ = load_eval_set(cfg["paths"]["eval_dir"])

    train_data = PairDataset(sp, train_en, train_gu, max_tokens=cfg["data"]["max_tokens_per_sent"])
    val_data = PairDataset(sp, val_en, val_gu)
    train_sampler = TokenBatchSampler(train_data.src_lens, train_data.tgt_lens, max_tokens, seed=seed)
    val_sampler = TokenBatchSampler(val_data.src_lens, val_data.tgt_lens, tcfg["max_tokens_per_gpu"], shuffle=False)
    workers = tcfg["num_workers"]
    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(val_data, batch_sampler=val_sampler, collate_fn=collate)
    print(f"Training pairs after length filter: {len(train_data):,}")
    print(f"Avg tokens  en: {train_data.src_lens.mean():.1f}  gu: {train_data.tgt_lens.mean():.1f}")
    print(f"Micro-batches/epoch: {len(train_sampler):,}  ->  updates/epoch: {len(train_sampler) // update_freq:,}")

    # ---- model, optimiser, schedule ----
    model = Transformer(**config).to(device)
    train_module = LossCompute(model, tcfg["label_smoothing"])
    if num_gpus > 1:
        train_module = nn.DataParallel(train_module)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["peak_lr"], betas=(0.9, 0.98), eps=1e-9, fused=device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def lr_at(step):
        """Linear warmup, then inverse square-root decay (Vaswani et al.)."""
        step = max(step, 1)
        return tcfg["peak_lr"] * min(step / tcfg["warmup_steps"], (tcfg["warmup_steps"] / step) ** 0.5)

    step, start_epoch, best_val_loss, history = 0, 0, float("inf"), []
    if tcfg.get("resume_checkpoint"):
        state = torch.load(tcfg["resume_checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        step, start_epoch = state["step"], state["epoch"] + 1
        best_val_loss, history = state["best_val_loss"], state["history"]
        print(f"Resumed from step {step}, epoch {start_epoch}, best val loss {best_val_loss:.4f}")
        del state
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    if not tcfg.get("resume_checkpoint"):  # don't let a previous run's files leak into this one's selection
        for stale in [*snapshot_dir.glob("snapshot_*.pt"), out_dir / "best.pt"]:
            stale.unlink(missing_ok=True)

    def save_training_state(epoch):
        atomic_save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "step": step,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "history": history,
                "config": config,
                "spm_model": spm_tensor(sp),
            },
            out_dir / "last.pt",
        )

    # ---- training loop ----
    quick_val = tcfg["quick_val_size"]
    train_start = time.time()
    deadline = train_start + tcfg["time_budget_hours"] * 3600
    snapshots, bad_evals, stop_reason, epoch = deque(), 0, None, start_epoch

    print(f"Training for up to {tcfg['max_epochs']} epochs / {tcfg['time_budget_hours']}h ...")
    for epoch in range(start_epoch, tcfg["max_epochs"]):
        train_sampler.set_epoch(epoch)
        train_module.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, running_count, micro = torch.zeros((), device=device), 0, 0
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", mininterval=10)

        for src, tgt_in, tgt_out in pbar:
            src = src.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)

            loss_sum, ntokens = train_module(src, tgt_in, tgt_out)
            loss = loss_sum.sum() / ntokens.sum()
            scaler.scale(loss / update_freq).backward()
            running_loss += loss.detach()
            running_count += 1
            micro += 1
            if micro % update_freq:
                continue

            step += 1
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0 and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    step=step, loss=f"{(running_loss / running_count).item():.3f}", lr=f"{lr_at(step):.2e}"
                )
                running_loss.zero_()
                running_count = 0

            if step % tcfg["eval_every"] == 0:
                val_loss = validation_loss(model, val_loader, device)
                quick = score_translations(model, sp, val_en[:quick_val], val_gu[:quick_val], device)
                train_module.train()
                hours = (time.time() - train_start) / 3600
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_bleu": quick["bleu"],
                        "val_chrf": quick["chrf++"],
                        "hours": hours,
                    }
                )
                improved = val_loss < best_val_loss
                print(
                    f"\n[step {step:>6} | epoch {epoch + 1} | {hours:.2f}h] val loss {val_loss:.4f} "
                    f"(ppl {np.exp(val_loss):.1f}) | greedy BLEU {quick['bleu']:.2f} "
                    f"chrF++ {quick['chrf++']:.2f}{'  * best' if improved else ''}"
                )
                print(f"    {val_en[0]}\n    -> {quick['hypotheses'][0]}")

                # Rolling weight snapshots for checkpoint averaging
                path = snapshot_dir / f"snapshot_{step}.pt"
                atomic_save(model.state_dict(), path)
                snapshots.append(path)
                while len(snapshots) > tcfg["num_snapshots_to_average"]:
                    os.remove(snapshots.popleft())

                if improved:
                    best_val_loss, bad_evals = val_loss, 0
                    save_for_app(
                        out_dir / "best.pt",
                        config,
                        model.state_dict(),
                        sp,
                        {"checkpoint": f"step {step}", "train_updates": step, "val_loss": round(val_loss, 4)},
                    )
                else:
                    bad_evals += 1
                save_training_state(epoch)

                if bad_evals >= tcfg["patience"]:
                    stop_reason = f"early stopping ({tcfg['patience']} evaluations without improvement)"
            if time.time() > deadline:
                stop_reason = stop_reason or "time budget reached"
            if stop_reason:
                break

        print(f"Epoch {epoch + 1} finished in {(time.time() - epoch_start) / 60:.1f} min")
        if stop_reason:
            break
    else:
        stop_reason = "max epochs reached"

    save_training_state(epoch)
    (out_dir / "history.json").write_text(
        json.dumps({"step": step, "stop_reason": stop_reason, "history": history}, indent=2), encoding="utf-8"
    )
    print(f"Stopped: {stop_reason}. Total updates: {step:,}")
    return {"step": step, "stop_reason": stop_reason}
