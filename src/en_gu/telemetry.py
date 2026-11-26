"""Privacy-preserving, versioned monitoring events for local logs and Cloud Logging."""

import hashlib
import hmac
import json
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache

from langdetect import PROFILES_DIRECTORY, DetectorFactory, LangDetectException

from en_gu.textutils import normalize_text, split_sentences

INPUT_FEATURES = (
    "input_chars",
    "input_tokens",
    "n_sentences",
    "byte_or_unknown_share",
    "digit_or_punctuation_share",
    "non_latin_share",
    "non_english",
)
OUTPUT_FEATURES = (
    "output_tokens",
    "output_input_ratio",
    "output_latin_share",
    "repetition_rate",
    "hit_length_limit",
    "mean_log_probability",
)


def ratio(count, total):
    return count / total if total else 0.0


def latin_share(text):
    letters = [c for c in text if c.isalpha()]
    return ratio(sum("LATIN" in unicodedata.name(c, "") for c in letters), len(letters))


@lru_cache(maxsize=1)
def language_factory():
    factory = DetectorFactory()
    factory.load_profile(PROFILES_DIRECTORY)
    factory.set_seed(0)
    return factory


def non_english(text):
    if sum(c.isalpha() for c in text) < 12:
        return None
    detector = language_factory().create()
    detector.append(text)
    try:
        return int(detector.detect() != "en")
    except LangDetectException:
        return None


def input_features(text, sp, tokens=None):
    if tokens is None:
        sentences = split_sentences(text)[1]
        tokens = [sp.encode(normalize_text(sentence)) for sentence in sentences]
    n_tokens = sum(map(len, tokens))
    letters = any(c.isalpha() for c in text)
    return {
        "input_chars": len(text),
        "input_tokens": n_tokens,
        "n_sentences": len(tokens),
        "byte_or_unknown_share": ratio(
            sum(sp.is_byte(token) or sp.is_unknown(token) for ids in tokens for token in ids), n_tokens
        ),
        "digit_or_punctuation_share": ratio(
            sum(c.isdigit() or unicodedata.category(c).startswith("P") for c in text), len(text)
        ),
        "non_latin_share": 1 - latin_share(text) if letters else 0.0,
        "non_english": non_english(text),
    }


def output_features(text, traces, input_tokens):
    decoded = [trace.decoded for trace in traces]
    count = sum(len(sentence.ids) for sentence in decoded)
    repeats, ngrams = 0, 0
    weighted_logp, scored_tokens = 0.0, 0
    for sentence in decoded:
        ids = sentence.ids
        trigrams = list(zip(ids, ids[1:], ids[2:]))
        repeats += len(trigrams) - len(set(trigrams))
        ngrams += len(trigrams)
        if sentence.mean_log_probability is not None:
            weight = len(ids) + (not sentence.hit_length_limit)
            scored_tokens += weight
            weighted_logp += sentence.mean_log_probability * weight
    return {
        "output_tokens": count,
        "output_input_ratio": ratio(count, input_tokens),
        "output_latin_share": latin_share(text),
        "repetition_rate": ratio(repeats, ngrams),
        "hit_length_limit": int(any(sentence.hit_length_limit for sentence in decoded)),
        "mean_log_probability": weighted_logp / scored_tokens if scored_tokens else None,
    }


def hash_input(text, key):
    return hmac.new(key, text.encode("utf-8"), hashlib.sha256).hexdigest()


def emit(event, **fields):
    # Cloud Run parses single-line JSON on stdout into jsonPayload. Keep only allowlisted fields at call sites.
    print(
        json.dumps(
            {"event": event, "schema": 1, "timestamp": datetime.now(timezone.utc).isoformat(), **fields},
            allow_nan=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
