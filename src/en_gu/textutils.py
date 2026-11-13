"""Small text helpers shared by training, evaluation and serving."""

import re


def normalize_text(text):
    """Collapse whitespace; SentencePiece handles Unicode normalisation."""
    return re.sub(r"\s+", " ", text).strip()
