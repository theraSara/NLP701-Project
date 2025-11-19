import torch 
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
from tqdm import tqdm


@torch.no_grad()
def comprehensiveness(df, tokenizer, model, get_importance_fn, ratios, device="cuda", max_length=512):
    """
    df: DataFrame with column 'texts'
    For each text: delete top-k% tokens (per ratio) and measure confidence drop.
    Returns DataFrame: columns ['ratio', 'comp_mean']
    """
    results = {r: [] for r in ratios}
    n_samples = len(df)

    for i, row in tqdm(df.iterrows(), total=n_samples):
        text = row["texts"]

        # importance on original
        try:
            tokens, scores = get_importance_fn(text, tokenizer, model, device=device)
        except Exception:
            continue

        # original confidence
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            logits = model(**inputs).logits
            probs = softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).item()
            conf_orig = probs[0, pred].item()
        except Exception:
            continue

        # rank tokens once
        seq_len = len(tokens)
        ignore_tokens = {"[CLS]", "[SEP]"}
        valid_idxs = [i for i, tok in enumerate(tokens) if tok not in ignore_tokens]
        ranked = sorted(valid_idxs, key=lambda i: scores[i], reverse=True)

        # per-ratio delete and re-score
        for r in ratios:
            try:
                k = max(1, int(r * seq_len))
                to_remove = set(ranked[:k])
                kept_tokens = [tokens[i] for i in range(seq_len) if i not in to_remove]
                new_text = tokenizer.convert_tokens_to_string(kept_tokens)

                new_inputs = tokenizer(new_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
                new_logits = model(**new_inputs).logits
                new_probs = softmax(new_logits, dim=-1)
                conf_new = new_probs[0, pred].item()

                comp = conf_orig - conf_new
                results[r].append(comp)
            except Exception:
                continue

    mean_comp = {r: np.nanmean(results[r]) for r in ratios}
    return pd.DataFrame({"ratio": ratios, "comp_mean": [mean_comp[r] for r in ratios]})

@torch.no_grad()
def sufficiency(df, tokenizer, model, get_importance_fn, ratios, device="cpu", max_length=512):
    """
    df: DataFrame with column 'texts'
    For each text, compute mean Sufficiency for each ratio.

    Sufficiency = drop in model confidence when keeping only the top-k% most important tokens.
    """
    results = {r: [] for r in ratios}
    n_samples = len(df)

    for i, row in tqdm(df.iterrows(), total=n_samples):
        text = row["texts"]

        # 1. Get token importance
        try:
            tokens, scores = get_importance_fn(text, tokenizer, model, device=device)
        except Exception:
            continue

        # 2. Get baseline confidence
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            logits = model(**inputs).logits
            probs = softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).item()
            conf_orig = probs[0, pred].item()
        except Exception:
            continue

        # 3. Pre-rank tokens (ignore special tokens)
        seq_len = len(tokens)
        ignore_tokens = {"[CLS]", "[SEP]"}
        valid_idxs = [i for i, tok in enumerate(tokens) if tok not in ignore_tokens]
        ranked = sorted(valid_idxs, key=lambda i: scores[i], reverse=True)

        # 4. Evaluate sufficiency at each ratio
        for r in ratios:
            try:
                k = max(1, int(r * seq_len))
                to_keep = set(ranked[:k])
                kept_tokens = [tokens[i] for i in range(seq_len) if i in to_keep or tokens[i] in ignore_tokens]
                new_text = tokenizer.convert_tokens_to_string(kept_tokens)

                new_inputs = tokenizer(new_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
                new_logits = model(**new_inputs).logits
                new_probs = softmax(new_logits, dim=-1)
                conf_new = new_probs[0, pred].item()

                # Sufficiency = confidence drop when only top-k tokens are kept
                suff = conf_orig - conf_new
                results[r].append(suff)
            except Exception:
                continue

    mean_suff = {r: np.nanmean(results[r]) for r in ratios}
    return pd.DataFrame({"ratio": ratios, "suff_mean": [mean_suff[r] for r in ratios]})