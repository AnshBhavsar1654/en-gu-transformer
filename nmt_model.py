"""Transformer encoder-decoder for English -> Gujarati translation.

This file is shared by the Kaggle training notebook (which writes it with
%%writefile) and the Streamlit app, so training and inference always use the
exact same architecture and decoding code.
"""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# These ids are baked into the SentencePiece model trained in the notebook
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def normalize_text(text):
    """Collapse whitespace; SentencePiece handles Unicode normalisation."""
    return re.sub(r"\s+", " ", text).strip()


def sinusoidal_positions(max_len, d_model):
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model): broadcasts over the batch dimension


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        self.w_q = nn.Linear(d_model, d_model)
        self.w_kv = nn.Linear(d_model, 2 * d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, memory, mask=None, is_causal=False):
        batch_size, q_len, d_model = query.shape
        k_len = memory.size(1)
        q = self.w_q(query).view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
        k, v = self.w_kv(memory).view(batch_size, k_len, 2, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
        # Fused attention kernel (Flash / memory-efficient) when the GPU supports it
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        return self.w_o(out.transpose(1, 2).reshape(batch_size, q_len, d_model))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


# Pre-LayerNorm layers: far more stable to train than the original post-LN design
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attention(h, h, src_mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask):
        # Target padding only ever sits after real tokens, so the causal mask is enough
        h = self.norm1(x)
        x = x + self.dropout(self.self_attention(h, h, is_causal=True))
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attention(h, memory, src_mask))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, dropout=0.1, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embed_scale = math.sqrt(d_model)

        # One joint EN+GU vocabulary, so a single embedding matrix is shared by the
        # encoder input, decoder input and output projection (fewer params, better generalisation)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.register_buffer("pe", sinusoidal_positions(max_len, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self._reset_parameters()
        self.output_proj.weight = self.embedding.weight

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[PAD_IDX].zero_()

    def embed(self, tokens):
        x = self.embedding(tokens) * self.embed_scale + self.pe[: tokens.size(1)]
        return self.dropout(x)

    def encode(self, src):
        src_mask = (src != PAD_IDX)[:, None, None, :]  # (B, 1, 1, S), True = attend
        x = self.embed(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x), src_mask

    def decode(self, tgt, memory, src_mask):
        x = self.embed(tgt)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask)
        return self.output_proj(self.decoder_norm(x))

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        return self.decode(tgt, memory, src_mask)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

BANNED_TOKENS = [PAD_IDX, UNK_IDX, BOS_IDX]


def _max_new_tokens(model, src, max_len_a, max_len_b):
    src_len = int((src != PAD_IDX).sum(1).max())
    return min(model.max_len - 1, int(max_len_a * src_len + max_len_b))


def _strip(ids):
    if EOS_IDX in ids:
        ids = ids[: ids.index(EOS_IDX)]
    return [i for i in ids if i != PAD_IDX]


@torch.no_grad()
def greedy_decode(model, src, max_len_a=1.5, max_len_b=10):
    max_new = _max_new_tokens(model, src, max_len_a, max_len_b)
    memory, src_mask = model.encode(src)
    ys = torch.full((src.size(0), 1), BOS_IDX, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_new):
        logits = model.decode(ys, memory, src_mask)[:, -1]
        logits[:, BANNED_TOKENS] = float("-inf")
        next_tok = logits.argmax(-1).masked_fill(finished, PAD_IDX)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished |= next_tok == EOS_IDX
        if finished.all():
            break
    return [_strip(seq) for seq in ys[:, 1:].tolist()]


@torch.no_grad()
def beam_search(model, src, beam_size=5, length_penalty=1.0, max_len_a=1.5, max_len_b=10):
    batch_size, k = src.size(0), beam_size
    device = src.device
    max_new = _max_new_tokens(model, src, max_len_a, max_len_b)

    memory, src_mask = model.encode(src)
    memory = memory.repeat_interleave(k, dim=0)
    src_mask = src_mask.repeat_interleave(k, dim=0)

    seqs = torch.full((batch_size * k, 1), BOS_IDX, dtype=torch.long, device=device)
    scores = torch.full((batch_size, k), float("-inf"), device=device)
    scores[:, 0] = 0.0  # all beams start identical, so only one is alive at the first step
    beam_offset = (torch.arange(batch_size, device=device) * k).unsqueeze(1)
    finished = [[] for _ in range(batch_size)]  # (normalised score, token ids)
    done = [False] * batch_size

    for _ in range(max_new):
        logits = model.decode(seqs, memory, src_mask)[:, -1].float()
        logp = F.log_softmax(logits, dim=-1)
        logp[:, BANNED_TOKENS] = float("-inf")
        vocab = logp.size(-1)

        candidates = (scores.view(-1, 1) + logp).view(batch_size, k * vocab)
        top_scores, top_idx = candidates.topk(2 * k, dim=1)
        top_beam = torch.div(top_idx, vocab, rounding_mode="floor").tolist()
        top_tok = (top_idx % vocab).tolist()
        top_scores = top_scores.tolist()

        next_scores = [[float("-inf")] * k for _ in range(batch_size)]
        next_beam = [[0] * k for _ in range(batch_size)]
        next_tok = [[PAD_IDX] * k for _ in range(batch_size)]
        for b in range(batch_size):
            if done[b]:
                continue
            n = 0
            for score, beam, tok in zip(top_scores[b], top_beam[b], top_tok[b]):
                if score == float("-inf"):
                    break
                if tok == EOS_IDX:
                    hyp = seqs[b * k + beam, 1:].tolist()
                    finished[b].append((score / (len(hyp) + 1) ** length_penalty, hyp))
                else:
                    next_scores[b][n], next_beam[b][n], next_tok[b][n] = score, beam, tok
                    n += 1
                    if n == k:
                        break
            if len(finished[b]) >= k or n == 0:
                done[b] = True
        if all(done):
            break

        select = (beam_offset + torch.tensor(next_beam, device=device)).view(-1)
        seqs = torch.cat([seqs[select], torch.tensor(next_tok, device=device).view(-1, 1)], dim=1)
        scores = torch.tensor(next_scores, device=device)

    results = []
    for b in range(batch_size):
        if not finished[b]:  # hit the length limit: fall back to the alive beams
            for j in range(k):
                score = scores[b, j].item()
                if score > float("-inf"):
                    hyp = seqs[b * k + j, 1:].tolist()
                    finished[b].append((score / max(len(hyp), 1) ** length_penalty, hyp))
        results.append(max(finished[b])[1] if finished[b] else [])
    return results


@torch.no_grad()
def translate(model, sp, sentences, beam_size=5, batch_size=32, max_len_a=1.5, max_len_b=10):
    """Translate a list of English sentences. `sp` is the SentencePiece processor."""
    model.eval()
    device = next(model.parameters()).device
    results = [""] * len(sentences)
    encoded = {}
    for i, sentence in enumerate(sentences):
        sentence = normalize_text(sentence)
        if sentence:
            encoded[i] = sp.encode(sentence)[: model.max_len - 1] + [EOS_IDX]

    order = sorted(encoded, key=lambda i: len(encoded[i]))  # similar lengths -> less padding
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        width = max(len(encoded[i]) for i in idx)
        src = torch.full((len(idx), width), PAD_IDX, dtype=torch.long)
        for row, i in enumerate(idx):
            src[row, : len(encoded[i])] = torch.tensor(encoded[i])
        src = src.to(device)
        if beam_size > 1:
            outputs = beam_search(model, src, beam_size, max_len_a=max_len_a, max_len_b=max_len_b)
        else:
            outputs = greedy_decode(model, src, max_len_a=max_len_a, max_len_b=max_len_b)
        for i, ids in zip(idx, outputs):
            results[i] = sp.decode(ids)
    return results


def load_for_inference(path, device="cpu"):
    """Load the exported checkpoint (config + weights + SentencePiece model in one file)."""
    import sentencepiece as spm

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(**ckpt["config"])
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # The output projection is tied to the embedding, so the export leaves it out
    if unexpected or set(missing) - {"output_proj.weight"}:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing}, unexpected: {unexpected}")
    model.to(device).eval()
    sp = spm.SentencePieceProcessor(model_proto=ckpt["spm_model"].numpy().tobytes())
    return model, sp, ckpt.get("metrics", {})
