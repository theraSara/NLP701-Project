import json
import pandas as pd
import numpy as np 

import torch
from torch.nn.functional import softmax

from typing import Iterable, Dict, Optional, Tuple, List

from lime.lime_text import LimeTextExplainer
from captum.attr import LayerIntegratedGradients


# === ATTENTION METHOD ===
@torch.no_grad()
def get_attention_score(
    text: str,
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 512,
) -> Tuple[List[str], List[float]]:
    """
    Return token-level importance scores based on mean attention
    across all layers and heads (CLS → token row).

    Returns:
        tokens: list of tokens (includes [CLS], [SEP])
        scores: list of normalized importance values (sum = 1)
    """
    model.to(device).eval()

    # tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # forward with attentions
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # list of [B, H, S, S]

    # average over layers and heads → [S, S]
    att = torch.stack(attentions, dim=0).mean(dim=(0, 2))[0]
    att = att / (att.sum(dim=-1, keepdim=True) + 1e-12)

    # find CLS (usually index 0)
    input_ids = inputs["input_ids"][0]
    cls_id = tokenizer.cls_token_id
    seq_len = int(inputs["attention_mask"][0].sum())
    cls_pos = 0
    if cls_id is not None:
        where = (input_ids[:seq_len] == cls_id).nonzero(as_tuple=False)
        if len(where) > 0:
            cls_pos = int(where[0].item())

    # CLS → token row
    cls_row = att[cls_pos, :seq_len]
    scores = (cls_row / (cls_row.sum() + 1e-12)).tolist()

    # tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[:seq_len])

    return tokens, scores

#############################################################################################################

## === LIME METHOD ===
@torch.no_grad()
def _predict_proba(texts, tokenizer, model, device="cpu", max_length=512):
    enc = tokenizer(list(texts),
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length).to(device)
    return softmax(model(**enc).logits, dim=-1).cpu().numpy()

def get_lime_score(
    text,
    tokenizer,
    model,
    device=device,
    max_length=256,
    num_samples=100,
    num_features=None,
    random_state=42,
    use_abs=True,
    l1_normalize=True,
):
    """
    Returns (tokens_without_spaces, scores) aligned to LIME's *non-space* tokens.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    base = _predict_proba([text], tokenizer, model, device, max_length)[0]
    target_label = int(np.argmax(base))

    if num_features is None:
        num_features = min(len(text.split()), 1000)

    explainer = LimeTextExplainer(random_state=random_state)
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda xs: _predict_proba(xs, tokenizer, model, device, max_length),
        labels=(target_label,),
        num_features=int(num_features),
        num_samples=num_samples,
    )

    # ---- raw tokens from LIME (may include spaces) ----
    dm = exp.domain_mapper
    idx_str = getattr(dm, "indexed_string", None)
    if idx_str is None:
        # fallback: use as_list; this returns only selected features, not full vector
        pairs = exp.as_list(label=target_label)
        toks = [t for t, _ in pairs]
        vals = [abs(w) if use_abs else float(w) for _, w in pairs]
        if l1_normalize and sum(vals) > 0:
            s = sum(vals); vals = [v/s for v in vals]
        return toks, vals

    raw = idx_str.as_list if isinstance(getattr(idx_str, "as_list", None), list) else idx_str.as_list()
    raw_tokens = list(raw)

    # ---- map non-space feature indices -> raw token indices ----
    nonspace_to_raw = []
    for i, tok in enumerate(raw_tokens):
        if not tok.isspace():
            nonspace_to_raw.append(i)

    # ---- LIME weights are given over non-space token indices ----
    index_weights = dict(exp.as_map()[target_label])  # {nonspace_idx: weight}
    raw_scores = np.zeros(len(raw_tokens), dtype=float)
    for nz_idx, w in index_weights.items():
        if 0 <= nz_idx < len(nonspace_to_raw):
            raw_idx = nonspace_to_raw[nz_idx]
            raw_scores[raw_idx] = abs(w) if use_abs else float(w)

    # ---- drop space tokens; keep only real tokens ----
    tokens, scores = [], []
    for tok, sc in zip(raw_tokens, raw_scores):
        if not tok.isspace():
            tokens.append(tok)
            scores.append(float(sc))

    # optional L1 normalization
    if l1_normalize:
        s = sum(scores)
        if s > 0:
            scores = [v / s for v in scores]
    return tokens, scores

#############################################################################################################

## === INTEGRATED GRADIENTS METHOD ===

@torch.no_grad()
def get_ig_score(
    text,
    tokenizer,
    model,
    device=device,
    max_length=256,
    num_samples=100,
    num_features=None,
    random_state=42,
    use_abs=True,
    l1_normalize=True,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return tokens, scores, 


#############################################################################################################

## === SHAP METHOD ===

@torch.no_grad()
def get_shap_score():
    return None