"""Small text helpers shared by training, evaluation and serving."""

import re

# A sentence ends at . ! or ? followed by whitespace. Simple on purpose: the model was trained on single
# sentences, so anything longer is split here and the pieces are translated independently.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text):
    """Collapse whitespace; SentencePiece handles Unicode normalisation."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    """Split text into sentences, remembering the paragraph layout.

    Returns `(layout, sentences)`: `layout[i]` is how many sentences paragraph i (a line of `text`) holds, so
    `join_sentences(layout, translations)` rebuilds the same line structure from the translated sentences.
    """
    layout, sentences = [], []
    for paragraph in text.split("\n"):
        parts = [s for s in SENTENCE_END.split(paragraph.strip()) if s]
        layout.append(len(parts))
        sentences.extend(parts)
    return layout, sentences


def join_sentences(layout, sentences):
    """Inverse of `split_sentences`: sentences of a paragraph are joined by spaces, paragraphs by newlines."""
    if sum(layout) != len(sentences):
        raise ValueError(f"layout expects {sum(layout)} sentences, got {len(sentences)}")
    it = iter(sentences)
    return "\n".join(" ".join(next(it) for _ in range(n)) for n in layout)
