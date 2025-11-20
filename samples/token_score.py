import json
import pandas as pd
import numpy as np 

import torch
from torch.nn.functional import softmax

from typing import Iterable, Dict, Optional, Tuple, List

from lime.lime_text import LimeTextExplainer
from captum.attr import LayerIntegratedGradients
import shap

from utils import special_token_set, filter_specials 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(scores):
    scores = np.array(scores, dtype=float)
    # absolute 
    scores = np.abs(scores)
    # l1
    scores = scores / (scores.sum() + 1e-12)
    return scores

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
    raw_scores = cls_row.tolist()
    scores = normalize(cls_row).tolist()

    # tokens
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[:seq_len])

    specials = special_token_set(tokenizer) 
    tokens, scores = filter_specials(raw_tokens, scores, specials)

    return tokens, scores, raw_scores, raw_tokens

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
    random_state=42):
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

    dm = exp.domain_mapper
    idx_str = getattr(dm, "indexed_string", None)
    if idx_str is None:
        pairs = exp.as_list(label=target_label)
        toks = [t for t, _ in pairs]
        vals = [w for _, w in pairs]
        scores = normalize(vals)  
        return toks, scores.tolist()

    raw_tokens = list(idx_str.as_list() if callable(getattr(idx_str, "as_list", None))
                      else idx_str.as_list)
    index_weights = dict(exp.as_map()[target_label])
    raw_scores = np.zeros(len(raw_tokens), dtype=float)

    nonspace_to_raw = [i for i, t in enumerate(raw_tokens) if not t.isspace()]
    for nz_idx, w in index_weights.items():
        if 0 <= nz_idx < len(nonspace_to_raw):
            raw_idx = nonspace_to_raw[nz_idx]
            raw_scores[raw_idx] = w

    tokens, scores = [], []
    for tok, sc in zip(raw_tokens, raw_scores):
        if not tok.isspace():
            tokens.append(tok)
            scores.append(sc)
    
    normalized_scores = normalize(scores).tolist()

    specials = special_token_set(tokenizer) 
    tokens_n, scores_n = filter_specials(tokens, normalized_scores, specials)

    return tokens_n, scores_n, scores, tokens

#############################################################################################################

# ## === INTEGRATED GRADIENTS METHOD ===

# # key (id(model), id(tokenizer), use_token_type) -> LayerIntegratedGradients
# IG_CACHE = {}

# def get_embed_layer(model):
#     if hasattr(model, "bert"):
#         return model.bert.embeddings
#     if hasattr(model, "distilbert"):
#         return model.distilbert.embeddings
#     if hasattr(model, "albert"):
#         return model.albert.embeddings
#     emb = model.get_input_embeddings()
#     if emb is None:
#         raise RuntimeError("Could not find embedding layer on this model.")
#     return emb

# def make_forward_fn(model, use_token_type):
#     # forward returning logits
#     # Captum will call it many times
#     if use_token_type:
#         def forward_func(input_ids, attention_mask, token_type_ids):
#             return model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids
#             ).logits
#     else:
#         def forward_func(input_ids, attention_mask):
#             return model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             ).logits
#     return forward_func

# def build_lig(model, tokenizer, use_token_type):
#     key = (id(model), id(tokenizer), bool(use_token_type))
#     lig = IG_CACHE.get(key)
#     if lig is not None:
#         return lig
#     layer = get_embed_layer(model)
#     forward = make_forward_fn(model, use_token_type)
#     lig = LayerIntegratedGradients(forward, layer)
#     IG_CACHE[key] = lig
#     return lig

# def get_ig_score(
#     text,
#     tokenizer,
#     model,
#     device=device,
#     max_length=256,
#     n_steps=16, # lower=faster
#     internal_batch_size=None,
#     use_abs=True,
#     l1_normalize=True,
# ):
#     model.to(device).eval()

#     enc = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=max_length,
#         padding="max_length",  # pad to max_length instead of True
#     ).to(device)

#     input_ids = enc["input_ids"]      # (1, max_length)
#     attention_mask = enc["attention_mask"]  # (1, max_length)
#     token_type_ids = enc.get("token_type_ids", None)
#     seq_len = int(attention_mask[0].sum().item())

#     # predicted target label -> without grad
#     with torch.no_grad():
#         logits = model(**enc).logits  # shape: (batch_size, num_classes)
#         # pick the predicted class for each sample
#         target_idx = torch.argmax(logits, dim=-1)  # shape: (batch_size,)


#     # build PAD baseline of the same shape
#     baseline = input_ids.clone()
#     special_ids = {
#         t for t in [
#             getattr(tokenizer, "cls_token_id", None),
#             getattr(tokenizer, "sep_token_id", None),
#             getattr(tokenizer, "pad_token_id", None),
#             getattr(tokenizer, "bos_token_id", None),
#             getattr(tokenizer, "eos_token_id", None),
#         ] if t is not None
#     }
#     pad_id = tokenizer.pad_token_id or 0
#     baseline[0] = torch.tensor([tid if tid in special_ids else pad_id for tid in input_ids[0].tolist()], device=device)

#     # Captum: build or retrieve LIG on (model, tokenizer, use_token_type)
#     use_token_type = token_type_ids is not None
#     lig = build_lig(model, tokenizer, use_token_type)

#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:seq_len])  # fallback
    
#     scores_list = []
#     try:
#         for i in range(len(target_idx)):
#             if use_token_type:
#                 atts = lig.attribute(
#                     inputs=input_ids,
#                     baselines=baseline,
#                     additional_forward_args=(attention_mask, token_type_ids),
#                     target=target_idx,
#                     n_steps=n_steps,
#                     internal_batch_size=internal_batch_size,
#                     return_convergence_delta=False,
#                 )
#             else:
#                 atts = lig.attribute(
#                     inputs=input_ids,
#                     baselines=baseline,
#                     additional_forward_args=(attention_mask,),
#                     target=target_idx,
#                     n_steps=n_steps,
#                     internal_batch_size=internal_batch_size,
#                     return_convergence_delta=False,
#                 )
#             scores_raw = atts.sum(dim=-1).squeeze(0)[:int(attention_mask[i].sum())].detach().cpu().numpy()
#             scores_proc = np.abs(scores_raw) if use_abs else scores_raw.copy()
#             if l1_normalize:
#                 s = np.sum(scores_proc)
#                 if s > 0:
#                     scores_proc = scores_proc / s
#             scores_list.append((tokens, scores_proc.tolist(), scores_raw.tolist()))

#     except Exception as e:
#         print(f"Attribution failed: {e}")
#         scores_raw = np.zeros(len(tokens))
#         scores_proc = np.zeros(len(tokens))


#     # # compute attributions (with auto_grad)
#     # scores_list = []
#     # for i in range(len(target_idx)):
#     #     atts = lig.attribute(
#     #         inputs=input_ids[i:i+1],
#     #         baselines=baseline[i:i+1],
#     #         additional_forward_args=(attention_mask[i:i+1], token_type_ids[i:i+1] if use_token_type else None),
#     #         target=int(target_idx[i]),
#     #         n_steps=n_steps,
#     #         internal_batch_size=internal_batch_size,
#     #         return_convergence_delta=False,
#     #     )
#     #     scores_raw = atts.sum(dim=-1).squeeze(0)[:int(attention_mask[i].sum())].detach().cpu().numpy()
#     #     scores_proc = np.abs(scores_raw) if use_abs else scores_raw.copy()
#     #     if l1_normalize:
#     #         s = np.sum(scores_proc)
#     #         if s > 0:
#     #             scores_proc = scores_proc / s
#     #     scores_list.append((tokens, scores_proc.tolist(), scores_raw.tolist()))

#     # tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:seq_len])
#     return tokens, scores_proc.tolist(), scores_raw.tolist()

# get_ig_score._needs_grad = True

# #############################################################################################################

# ## === SHAP METHOD ===

# @torch.no_grad()
# def _predict_proba_np(texts, tokenizer, model, device=device, max_length=256):
#     """
#     texts: list[str] -> np.ndarray of probs (N, C)
#     Safe wrapper used by SHAP for batching.
#     """
#     enc = tokenizer(
#         list(texts),
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=max_length
#     ).to(device)
#     logits = model(**enc).logits
#     return torch.softmax(logits, dim=-1).cpu().numpy()

# def get_shap_score(
#     text,
#     tokenizer,
#     model,
#     device,
#     max_length=256,
#     max_evals=100,
#     use_abs=True,
#     l1_normalize=True
# ):
#     model.to(device).eval()

#     # get the predictions class
#     base = _predict_proba_np([text], tokenizer, model)

#     def predict_proba_shap(texts):
#         # SHAP needs a list of texts to get the probabilities
#         if isinstance(texts, np.ndarray):
#             texts = texts.tolist()

#         # handle the empty or single strings
#         if isinstance(texts, str):
#             texts = [texts]

#         # tokenize batch
#         inputs = tokenizer(
#             texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=max_length
#         ).to(device)

#         # get predictions
#         with torch.no_grad():
#             logits = model(**inputs).logits
#             probs = softmax(logits, dim=-1)

#         return probs.cpu().numpy()
    
#     # SHAP explainer (tokenizer as the masker)
#     explainer = shap.Explainer(predict_proba_shap, tokenizer)

#     # generate SHAP values
#     try:
#         shap_values = explainer(
#             [text],
#             max_evals=max_evals,
#             silent=True 
#         )
#     except Exception as e: 
#         print(f"SHAP failed: {e}")
#         inputs = tokenizer(
#             text, 
#             return_tensors="pt", 
#             truncation=True, 
#             max_length=max_length
#         )
#         tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])




#     return None
