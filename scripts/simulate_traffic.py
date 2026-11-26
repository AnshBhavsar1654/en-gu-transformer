"""Replay normal and shifted traffic against a local or staging English-to-Gujarati API."""

import argparse
import random
import time
from collections import Counter
from pathlib import Path

from en_gu.smoke import request

PROFILES = ("normal", "long", "mixed", "non-english", "news", "conversation")
VARIATIONS = {
    "mixed": ["Can we meet આજે after lunch?", "Mujhe the meeting time batao.", "We will go to બજાર tomorrow."],
    "non-english": ["આજે વરસાદ પડશે અને રસ્તાઓ ભીના રહેશે.", "क्या तुम कल मेरे घर आओगे?", "Je prends le train demain."],
    "news": [
        "The central bank released revised inflation figures after its quarterly policy meeting.",
        "Auditors presented their annual public expenditure report to the parliamentary committee.",
    ],
    "conversation": ["Do you want to grab lunch later?", "Could you call me when you get home?", "I'll see you soon."],
}


def samples(profile, sources, count, seed):
    rng = random.Random(seed)
    short_sources = [line for line in sources if 35 <= len(line) <= 90]
    for _ in range(count):
        if profile == "normal":
            yield rng.choice(sources)
        elif profile == "long":
            sentences = []
            while len(" ".join(sentences)) < 850 and len(sentences) < 16:
                sentences.append(rng.choice(short_sources).rstrip(".!?") + ".")
            yield " ".join(sentences)
        else:
            yield rng.choice(VARIATIONS[profile])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--profile", choices=(*PROFILES, "all"), default="all")
    parser.add_argument("--count", type=int, default=20, help="requests per profile")
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sources", type=Path, default=Path("data/eval/val.en"))
    args = parser.parse_args()
    if args.count < 1 or args.delay < 0:
        parser.error("--count must be positive and --delay nonnegative")
    sources = [
        line.strip() for line in args.sources.read_text(encoding="utf-8").splitlines() if 0 < len(line.strip()) <= 700
    ]
    if not sources or (args.profile in ("long", "all") and not any(35 <= len(line) <= 90 for line in sources)):
        parser.error("--sources needs usable sentences, including 35–90 character inputs for the long profile")
    failures = 0
    for profile in PROFILES if args.profile == "all" else [args.profile]:
        statuses = Counter()
        for text in samples(profile, sources, args.count, args.seed):
            status, _, _ = request(
                args.url,
                "POST",
                "/translate",
                {"text": text, "beam_size": args.beam_size},
                headers={"User-Agent": "en-gu-simulated-traffic"},
                timeout=60,
            )
            statuses[status] += 1
            time.sleep(args.delay)
        print(f"Simulated {profile}: {dict(statuses)}")
        failures += args.count - statuses[200]
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
