import pandas as pd 
import math
import random
import re
from utils import set_seed, load_data

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

def get_shuffle_df(df, ratios, seed: int = 42, text_col: str = "texts", row_id_col: str | None = None,):

    out = df.copy()

    # base cleaned text once
    base = out[text_col].astype(str).map(normalize_spaces)

    # helper to get a deterministic seed for (row_id, ratio)
    def _seed_for(row_id: int | str, r: float) -> int:
        # mix seed, ratio (in integer hundredths), and row id deterministically
        r_key = int(round(r * 100))  # e.g. 5 for 0.05
        # simple integer mixing; avoids Python's randomized hash
        return (seed * 1_000_003) ^ (r_key * 97) ^ (int(row_id) * 1_001)

    # choose a stable row identifier
    row_ids = out[row_id_col] if (row_id_col and row_id_col in out.columns) else out.index

    for r in ratios:
        col = f"{r}"
        out[col] = [
            _shuffle_sentence_words(txt, r, random.Random(_seed_for(rid, r)))
            for txt, rid in zip(base.tolist(), row_ids.tolist())
        ]

    return out 

if __name__ == "__main__":
    set_seed()

    imdb_path = 'sampled/imdb_sampled_500.pkl'
    sst2_path = 'sampled/sst2_sampled_436.pkl'

    imdb_df = load_data(imdb_path).iloc[:100]
    sst2_df = load_data(sst2_path).iloc[:100]

    RATIOS = (0.01, 0.05, 0.1, 0.2, 0.5)

    imdb_df_shuffle = get_shuffle_df(imdb_df, ratios=RATIOS, row_id_col="indices")
    imdb_df_shuffle.to_csv("imdb_shuffled_100.csv", index=False)

    sst2_df_shuffle = get_shuffle_df(sst2_df, ratios=RATIOS, row_id_col="indices")
    sst2_df_shuffle.to_csv("sst2_shuffled_100.csv", index=False)

