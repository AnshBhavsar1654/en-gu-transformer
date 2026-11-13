"""Corpus cleaning, the frozen evaluation split, and the datasets / batching used for training."""

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from en_gu.model import BOS_IDX, EOS_IDX, PAD_IDX
from en_gu.textutils import normalize_text

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars are optional

    def tqdm(iterable, **_):
        return iterable


GUJARATI_RE = re.compile(r"[઀-૿]")
LATIN_RE = re.compile(r"[A-Za-z]")

EVAL_FILES = ("val.en", "val.gu", "test.en", "test.gu")
MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def keep_pair(en, gu, max_words=80):
    if not en or not gu or en == gu:
        return False
    n_en, n_gu = len(en.split()), len(gu.split())
    if n_en > max_words or n_gu > max_words:
        return False
    if max(n_en, n_gu) > 3 * min(n_en, n_gu) + 2:  # badly misaligned pair
        return False
    if GUJARATI_RE.search(en) or not LATIN_RE.search(en):
        return False
    gu_chars = len(GUJARATI_RE.findall(gu))
    return gu_chars > 0 and len(LATIN_RE.findall(gu)) <= 0.3 * gu_chars  # mostly Gujarati script


def load_clean_pairs(corpus_dir, max_sentences=None, max_words=80):
    """Read train.en / train.gu, normalise, filter and deduplicate. Returns two parallel lists."""
    en_lines, gu_lines, seen = [], [], set()
    total = 0
    corpus_dir = Path(corpus_dir)
    with (
        open(corpus_dir / "train.en", encoding="utf-8") as f_en,
        open(corpus_dir / "train.gu", encoding="utf-8") as f_gu,
    ):
        for en, gu in tqdm(zip(f_en, f_gu), desc="Reading corpus"):
            if max_sentences and total >= max_sentences:
                break
            total += 1
            en, gu = normalize_text(en), normalize_text(gu)
            if not keep_pair(en, gu, max_words):
                continue
            if (en, gu) in seen:
                continue
            seen.add((en, gu))
            en_lines.append(en)
            gu_lines.append(gu)
    if total:
        print(f"Kept {len(en_lines):,} of {total:,} pairs ({100 * len(en_lines) / total:.1f}%)")
    return en_lines, gu_lines


# ---------------------------------------------------------------------------
# Splits and the frozen evaluation set
# ---------------------------------------------------------------------------


def split_pairs(en, gu, seed, val_size, test_size, frozen=None):
    """Deterministic split. `frozen` = ((val_en, val_gu), (test_en, test_gu)) reuses an existing eval set.

    Training never contains a sentence whose (lower-cased) English source appears in val or test, so
    the scores are honest. Training pairs keep the permutation order.
    """
    perm = np.random.default_rng(seed).permutation(len(en))
    if frozen is None:
        test_idx = perm[:test_size]
        val_idx = perm[test_size : test_size + val_size]
        val = ([en[i] for i in val_idx], [gu[i] for i in val_idx])
        test = ([en[i] for i in test_idx], [gu[i] for i in test_idx])
    else:
        val, test = frozen
    heldout = {s.lower() for s in itertools.chain(val[0], test[0])}
    train_idx = [i for i in perm if en[i].lower() not in heldout]
    train = ([en[i] for i in train_idx], [gu[i] for i in train_idx])
    return {"train": train, "val": val, "test": test}


def write_lines(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_eval_set(eval_dir, val, test, meta):
    """Write val/test files plus a manifest of line counts and hashes that marks them as frozen."""
    eval_dir = Path(eval_dir)
    for name, (en, gu) in (("val", val), ("test", test)):
        write_lines(eval_dir / f"{name}.en", en)
        write_lines(eval_dir / f"{name}.gu", gu)
    files = {n: {"lines": len(read_lines(eval_dir / n)), "sha256": sha256_file(eval_dir / n)} for n in EVAL_FILES}
    with open(eval_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump({**meta, "files": files}, f, indent=2)


def load_eval_set(eval_dir):
    """Load the frozen eval set, refusing to continue if any file no longer matches its recorded hash."""
    eval_dir = Path(eval_dir)
    manifest_path = eval_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No frozen eval set in {eval_dir}: run `python -m en_gu prepare` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in EVAL_FILES:
        expected = manifest["files"][name]["sha256"]
        if sha256_file(eval_dir / name) != expected:
            raise RuntimeError(f"{eval_dir / name} was modified after it was frozen (hash mismatch).")
    val = (read_lines(eval_dir / "val.en"), read_lines(eval_dir / "val.gu"))
    test = (read_lines(eval_dir / "test.en"), read_lines(eval_dir / "test.gu"))
    return val, test


def has_frozen_eval(eval_dir):
    return (Path(eval_dir) / MANIFEST_NAME).exists()


# ---------------------------------------------------------------------------
# Tokenised datasets and token-based batching
# ---------------------------------------------------------------------------


def encode_corpus(sp, lines, chunk=200_000):
    """Tokenise into one flat int32 array + offsets (far less RAM than lists of lists)."""
    lengths, pieces = [], []
    for start in tqdm(range(0, len(lines), chunk), desc="Tokenising", leave=False):
        ids = sp.encode(lines[start : start + chunk])
        lengths.extend(len(x) for x in ids)
        pieces.append(np.fromiter(itertools.chain.from_iterable(ids), dtype=np.int32))
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32), offsets


class PairDataset(Dataset):
    def __init__(self, sp, en_lines, gu_lines, max_tokens=None):
        self.src, self.src_off = encode_corpus(sp, en_lines)
        self.tgt, self.tgt_off = encode_corpus(sp, gu_lines)
        src_lens = np.diff(self.src_off)
        tgt_lens = np.diff(self.tgt_off)
        keep = (src_lens > 0) & (tgt_lens > 0)
        if max_tokens:
            keep &= (src_lens + 1 <= max_tokens) & (tgt_lens + 1 <= max_tokens)
            keep &= (tgt_lens <= 3 * src_lens + 5) & (src_lens <= 3 * tgt_lens + 5)
        self.indices = np.nonzero(keep)[0]
        self.src_lens = src_lens[self.indices]
        self.tgt_lens = tgt_lens[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return (
            self.src[self.src_off[j] : self.src_off[j + 1]],
            self.tgt[self.tgt_off[j] : self.tgt_off[j + 1]],
        )


def collate(batch):
    src_width = max(len(s) for s, _ in batch) + 1
    tgt_width = max(len(t) for _, t in batch) + 1
    src = torch.full((len(batch), src_width), PAD_IDX, dtype=torch.long)
    tgt_in = torch.full((len(batch), tgt_width), PAD_IDX, dtype=torch.long)
    tgt_out = torch.full((len(batch), tgt_width), PAD_IDX, dtype=torch.long)
    for row, (s, t) in enumerate(batch):
        s, t = torch.from_numpy(s), torch.from_numpy(t)
        src[row, : len(s)] = s
        src[row, len(s)] = EOS_IDX
        tgt_in[row, 0] = BOS_IDX
        tgt_in[row, 1 : len(t) + 1] = t
        tgt_out[row, : len(t)] = t
        tgt_out[row, len(t)] = EOS_IDX
    return src, tgt_in, tgt_out


class TokenBatchSampler(Sampler):
    """Groups sentences of similar length so each batch holds ~max_tokens padded tokens."""

    def __init__(self, src_lens, tgt_lens, max_tokens, shuffle=True, seed=0):
        self.lens = (np.maximum(src_lens, tgt_lens) + 1).astype(np.int64)
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        if self.shuffle:  # sort by length, random order among equal lengths
            order = np.lexsort((rng.random(len(self.lens)), self.lens))
        else:
            order = np.argsort(self.lens, kind="stable")
        lens = self.lens[order].tolist()
        batches, batch = [], []
        for idx, length in zip(order.tolist(), lens):  # ascending, so `length` is the batch width
            if batch and length * (len(batch) + 1) > self.max_tokens:
                batches.append(batch)
                batch = []
            batch.append(idx)
        if batch:
            batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
