"""Joint English+Gujarati SentencePiece tokenizer."""

import os
from pathlib import Path

import sentencepiece as spm
import torch

from en_gu.config import artifact_path
from en_gu.data import read_lines
from en_gu.model import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX

SPM_NAME = "spm_en_gu"


def spm_model_path(cfg):
    return artifact_path(cfg, "tokenizer", f"{SPM_NAME}.model")


def load_tokenizer(path):
    return spm.SentencePieceProcessor(model_file=str(path))


def check_tokenizer(sp, vocab_size):
    if not (sp.get_piece_size() == vocab_size and sp.pad_id() == PAD_IDX and sp.eos_id() == EOS_IDX):
        raise RuntimeError(
            f"Tokenizer has {sp.get_piece_size()} pieces (pad={sp.pad_id()}, eos={sp.eos_id()}), "
            f"expected {vocab_size} pieces with pad={PAD_IDX}, eos={EOS_IDX}."
        )


def train_tokenizer(cfg):
    """One unigram model for both languages. Byte fallback means no token is ever <unk>."""
    tcfg, processed = cfg["tokenizer"], Path(cfg["paths"]["processed_dir"])
    out = spm_model_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)

    resume = cfg["train"].get("resume_checkpoint")
    if resume:  # reuse the exact tokenizer the checkpoint was trained with
        state = torch.load(resume, map_location="cpu", weights_only=True)
        out.write_bytes(state["spm_model"].numpy().tobytes())
        print(f"Loaded tokenizer from {resume}")
    else:
        sentences = read_lines(processed / "train.en") + read_lines(processed / "train.gu")
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(sentences),
            model_prefix=str(out.with_suffix("")),
            vocab_size=tcfg["vocab_size"],
            model_type="unigram",
            character_coverage=tcfg["character_coverage"],  # drops ultra-rare noise; byte fallback covers it
            byte_fallback=True,
            input_sentence_size=tcfg["input_sentence_size"],
            shuffle_input_sentence=True,
            pad_id=PAD_IDX,
            unk_id=UNK_IDX,
            bos_id=BOS_IDX,
            eos_id=EOS_IDX,
            num_threads=os.cpu_count(),
            minloglevel=2,
        )
    sp = load_tokenizer(out)
    check_tokenizer(sp, tcfg["vocab_size"])
    print(f"Tokenizer written to {out} ({sp.get_piece_size()} pieces)")
    return out
