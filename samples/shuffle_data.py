import math
import random
import re

def normalize_spaces(s: str) -> str:
    """Collapse all whitespace to single spaces and strip ends."""
    return re.sub(r"\s+", " ", (s or "")).strip()

def _shuffle_sentence_words(text: str, ratio: float, rng: random.Random) -> str:
    """Shuffle ceil(ratio*n) word positions in-place; others stay fixed."""
    text = normalize_spaces(text)
    if not text:
        return text
    words = text.split(" ")
    n = len(words)
    if n <= 1:
        return text

    k = max(1 if ratio > 0 else 0, min(n, math.ceil(ratio * n)))
    if k == 0:
        return text
    if k == n:
        rng.shuffle(words)
        return " ".join(words)

    idx = list(range(n))
    chosen = sorted(rng.sample(idx, k))
    chosen_words = [words[i] for i in chosen]
    rng.shuffle(chosen_words)
    for pos, w in zip(chosen, chosen_words):
        words[pos] = w
    return " ".join(words)

def get_shuffle(text: str, RATIOS, seed: int = 42) -> dict[str, str]:
    out = {"0": normalize_spaces(text)}
    for r in RATIOS:
        key = f"{r}"
        # stable per-ratio RNG (independent across ratios)
        rng = random.Random(seed * 1000 + int(round(r * 100)))
        out[key] = _shuffle_sentence_words(out["0"], r, rng)
    return out

