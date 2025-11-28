import json
import pandas as pd
import numpy as np 

import torch
from torch.nn.functional import softmax

from typing import Iterable, Dict, Optional, Tuple, List

from utils import filter_specials

from lime.lime_text import LimeTextExplainer
from captum.attr import LayerIntegratedGradients
import shap

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
    scores = cls_row.tolist()
    normalized_scores = normalize(cls_row).tolist()

    # tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[:seq_len])

    return tokens, normalized_scores, scores

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
    return tokens, normalized_scores, scores.tolist(),

#############################################################################################################

## === INTEGRATED GRADIENTS METHOD ===

# key (id(model), id(tokenizer), use_token_type) -> LayerIntegratedGradients
IG_CACHE = {}

def get_embed_layer(model):
    if hasattr(model, "bert"):
        return model.bert.embeddings
    if hasattr(model, "distilbert"):
        return model.distilbert.embeddings
    if hasattr(model, "albert"):
        return model.albert.embeddings
    emb = model.get_input_embeddings()
    if emb is None:
        raise RuntimeError("Could not find embedding layer on this model.")
    return emb

def make_forward_fn(model, use_token_type):
    # forward returning logits
    # Captum will call it many times
    if use_token_type:
        def forward_func(input_ids, attention_mask, token_type_ids):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).logits
    else:
        def forward_func(input_ids, attention_mask):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits
    return forward_func

def build_lig(model, tokenizer, use_token_type):
    key = (id(model), id(tokenizer), bool(use_token_type))
    lig = IG_CACHE.get(key)
    if lig is not None:
        return lig
    layer = get_embed_layer(model)
    forward = make_forward_fn(model, use_token_type)
    lig = LayerIntegratedGradients(forward, layer)
    IG_CACHE[key] = lig
    return lig

def get_ig_score(
    text,
    tokenizer,
    model,
    device=device,
    max_length=256,
    n_steps=16, # lower=faster
    internal_batch_size=None,
    use_abs=True,
    l1_normalize=True,
):
    model.to(device).eval()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",  # pad to max_length instead of True
    ).to(device)

    input_ids = enc["input_ids"]      # (1, max_length)
    attention_mask = enc["attention_mask"]  # (1, max_length)
    token_type_ids = enc.get("token_type_ids", None)
    seq_len = int(attention_mask[0].sum().item())

    # predicted target label -> without grad
    with torch.no_grad():
        logits = model(**enc).logits  # shape: (batch_size, num_classes)
        # pick the predicted class for each sample
        target_idx = torch.argmax(logits, dim=-1)  # shape: (batch_size,)


    # build PAD baseline of the same shape
    baseline = input_ids.clone()
    special_ids = {
        t for t in [
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
        ] if t is not None
    }
    pad_id = tokenizer.pad_token_id or 0
    baseline[0] = torch.tensor([tid if tid in special_ids else pad_id for tid in input_ids[0].tolist()], device=device)

    # Captum: build or retrieve LIG on (model, tokenizer, use_token_type)
    use_token_type = token_type_ids is not None
    lig = build_lig(model, tokenizer, use_token_type)

    tokens_raw = tokenizer.convert_ids_to_tokens(input_ids[0][:seq_len])  # fallback
    
    try:
        atts = lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask, token_type_ids) if use_token_type
                                    else (attention_mask,),
            target=target_idx,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=False,
        )

        scores_raw = (atts.sum(dim=-1).squeeze(0)[:seq_len].detach().cpu().numpy())
        scores_proc = np.abs(scores_raw) if use_abs else scores_raw.copy()
        if l1_normalize:
            s = np.sum(scores_proc)
            if s > 0:
                scores_proc = scores_proc / s
        tokens, scores = filter_specials(tokens_raw, scores_proc.tolist())

    except Exception as e:
        print(f"Attribution failed: {e}")
        scores_raw = np.zeros(len(tokens))
        scores_proc = np.zeros(len(tokens))

    return tokens, scores, scores_raw.tolist(), tokens_raw

get_ig_score._needs_grad = True

#############################################################################################################

## === SHAP METHOD ===

@torch.no_grad()
def get_shap_score(
    text,
    tokenizer,
    model,
    device,
    max_length = 256,
    max_evals=100,
    use_abs = True,
    l1_normalize= True,
):

    model.to(device).eval()
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    seq_len = len(tokens)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        pred_class = outputs.logits.argmax(dim=-1).item()
    
    try:
        # Create prediction function
        def predict_fn(texts):
            # SHAP may pass str, list of str, or np.ndarray
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, np.ndarray):
                texts = texts.flatten().tolist()
            elif isinstance(texts, list):
                # flatten list of lists
                if len(texts) > 0 and isinstance(texts[0], (list, np.ndarray)):
                    texts = [item for sublist in texts for item in (sublist.tolist() if isinstance(sublist, np.ndarray) else sublist)]
            else:
                texts = list(texts)

            batch_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            with torch.no_grad():
                logits = model(**batch_inputs).logits
                probs = torch.softmax(logits, dim=-1)

            return probs.cpu().numpy()

        
        # Create SHAP explainer
        explainer = shap.Explainer(predict_fn, tokenizer)
        
        # Get SHAP values
        shap_values = explainer(
            [text],
            max_evals=max_evals,
            silent=True
        )
        
        # Extract scores for predicted class
        if len(shap_values.values[0].shape) > 1:
            shap_scores = shap_values.values[0][:, pred_class]
        else:
            shap_scores = shap_values.values[0]
        
        # SHAP tokens (word-level)
        shap_tokens = shap_values.data[0]
        
        # Align SHAP scores to model tokens
        scores_raw = align_shap_simple(shap_tokens, shap_scores, tokens)
        
    except Exception as e:
        print(f"⚠️  SHAP failed on text (len={len(text)}): {str(e)[:100]}")
        print(f"   Falling back to masking-based importance")
        
        # Fallback: masking-based importance
        scores_raw = compute_masking_importance(
            text, tokenizer, model, device, max_length, pred_class
        )
    
    # Process scores
    scores_processed = np.abs(scores_raw) if use_abs else np.array(scores_raw).copy()
    scores_raw = np.array(scores_raw) 
    
    if l1_normalize:
        total = np.sum(np.abs(scores_processed))
        if total > 1e-10:
            scores_processed = scores_processed / total
        else:
            scores_processed = np.ones(len(scores_processed)) / len(scores_processed)

    return tokens, scores_processed.tolist(), scores_raw.tolist(), tokens

def align_shap_simple(shap_tokens, shap_scores, model_tokens):
    scores = np.zeros(len(model_tokens))
    
    # Clean tokens
    def clean(t):
        return t.replace('##', '').replace('▁', '').replace('Ġ', '').strip().lower()
    
    shap_words = [clean(t) for t in shap_tokens]
    model_words = [clean(t) for t in model_tokens]
    
    shap_idx = 0
    for i, model_word in enumerate(model_words):
        # Skip special tokens
        if model_tokens[i] in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            scores[i] = 0.0
            continue
        
        # Find matching SHAP word
        if shap_idx < len(shap_words):
            if model_word and shap_words[shap_idx] and (
                model_word in shap_words[shap_idx] or 
                shap_words[shap_idx] in model_word
            ):
                scores[i] = shap_scores[shap_idx]
            else:
                shap_idx += 1
                if shap_idx < len(shap_words):
                    scores[i] = shap_scores[shap_idx]
    
    return scores


def compute_masking_importance(text, tokenizer, model, device, max_length, pred_class):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    
    input_ids = inputs['input_ids'][0]
    seq_len = len(input_ids)
    
    # Baseline
    with torch.no_grad():
        baseline_logits = model(**inputs).logits[0, pred_class].item()
    
    # Mask each token
    scores = np.zeros(seq_len)
    mask_id = getattr(tokenizer, 'mask_token_id', tokenizer.pad_token_id or 0)
    
    for i in range(seq_len):
        masked_ids = input_ids.clone()
        masked_ids[i] = mask_id
        
        with torch.no_grad():
            masked_output = model(
                input_ids=masked_ids.unsqueeze(0),
                attention_mask=inputs['attention_mask']
            )
            masked_logits = masked_output.logits[0, pred_class].item()
        
        scores[i] = baseline_logits - masked_logits
    
    return scores