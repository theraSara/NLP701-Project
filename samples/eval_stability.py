import numpy as np 
import random 
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd


# --- helper: shuffle tokens with a given ratio ---
def shuffle_tokens(text: str, ratio: float) -> str:
    """
    Randomly shuffle a given ratio of tokens in the text.
    """
    tokens = text.split()
    n = len(tokens)
    k = int(ratio * n)
    if k < 2:
        return text  # nothing to shuffle

    idxs = list(range(n))
    shuffle_idxs = random.sample(idxs, k)
    shuffled_part = [tokens[i] for i in shuffle_idxs]
    random.shuffle(shuffled_part)
    for i, j in enumerate(shuffle_idxs):
        tokens[j] = shuffled_part[i]

    return " ".join(tokens)

# --- helper: compute stability (Spearman) between two vectors ---
def spearman_stability(scores1, scores2):
    # pad shorter one if tokenization length changes
    m = min(len(scores1), len(scores2))
    if m < 2:
        return np.nan
    corr, _ = spearmanr(scores1[:m], scores2[:m])
    return corr

# --- main stability evaluation ---
def stability(df, tokenizer, model, get_importance_fn, ratios, device="cpu"):
    """
    df: DataFrame with column 'texts'
    For each text: compute mean Spearman correlation for each shuffle ratio.
    """
    results = {r: [] for r in ratios}
    n_samples = len(df)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if i >= n_samples:
            break
        text = row["texts"]

        # 1. Get attention-based importance for original
        try:
            tokens, orig_scores, raw_scores = get_importance_fn(text, tokenizer, model, device=device)
        except Exception:
            continue

        # 2. For each ratio, create shuffled version and compare
        for r in ratios:
            shuffled_text = shuffle_tokens(text, r)
            try:
                _, shuf_scores, _ = get_importance_fn(shuffled_text, tokenizer, model, device=device)
            except Exception:
                continue
            corr = spearman_stability(orig_scores, shuf_scores)
            results[r].append(corr)

    # 3. Compute mean correlation per ratio
    mean_stability = {r: np.nanmean(results[r]) for r in ratios}
    return pd.DataFrame({"ratio": ratios, "mean_spearman": [mean_stability[r] for r in ratios]})