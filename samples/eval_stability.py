import numpy as np 
import random 
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd
import json
from pathlib import Path


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
def stability(df, tokenizer, model, get_importance_fn, ratios, device="cpu", save_log_to=None):
    """
    df: DataFrame with column 'texts'
    For each text: compute mean Spearman correlation for each shuffle ratio.
    """
    details = {}
    results_per_ratio = {r: [] for r in ratios}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row["texts"]
        sample_id = row.get("indices", i)
        details[str(sample_id)] = {
            "text": text,
            "label": row.get("label", None),
            "indices": sample_id
        }

        # 1. Get attention-based importance for original
        try:
            _, base_scores, _ = get_importance_fn(text, tokenizer, model, device=device)
        except Exception:
            continue

        # 2. For each ratio, create shuffled version and compare
        for r in ratios:
            shuffled_text = shuffle_tokens(text, r)
            try:
                _, shuf_scores, _ = get_importance_fn(shuffled_text, tokenizer, model, device=device)
            except Exception:
                continue

            corr = spearman_stability(base_scores, shuf_scores)
            results_per_ratio[r].append(corr)
            details[str(sample_id)][f"{r:.2f}"] = {
                "spearman": corr,
                "base_score": base_scores,
                "shuf_scores": shuf_scores,
                "len_orig": len(base_scores),
                "len_shuf": len(shuf_scores),
            }

    # 3. Compute mean correlation per ratio
    mean_stability = {r: float(np.nanmean(results_per_ratio[r])) if results_per_ratio[r] else np.nan for r in ratios}
    df_plot = pd.DataFrame({"ratio": ratios, "mean_spearman": [mean_stability[r] for r in ratios]})

    # save json
    if save_log_to is not None:
        Path(save_log_to).parent.mkdir(parents=True, exist_ok=True)
        with open(save_log_to, "w", encoding="utf-8") as f:
            json.dump({"by_id": details}, f, indent=2)
        print(f"(stability) Per-sample Spearman log saved to: {save_log_to}")

    return df_plot
