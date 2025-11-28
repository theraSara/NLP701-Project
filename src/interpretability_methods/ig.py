import numpy as np 

import torch

from utils import filter_specials, special_token_set, normalize

from captum.attr import LayerIntegratedGradients


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# Computes Integrated Gradients attribution scores:
def get_ig_score(
    text,
    tokenizer,
    model,
    device=device,
    max_length=256,
    n_steps=16, 
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
    # setting non-special tokens in the baseline to the PAD ID
    baseline[0] = torch.tensor([tid if tid in special_ids else pad_id for tid in input_ids[0].tolist()], device=device)

    # Captum: build or retrieve LIG on (model, tokenizer, use_token_type)
    use_token_type = token_type_ids is not None
    lig = build_lig(model, tokenizer, use_token_type)

    tokens_raw = tokenizer.convert_ids_to_tokens(input_ids[0][:seq_len])

    # get the set of special tokens by string for filtering the output
    specials = special_token_set(tokenizer)
    
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
            scores_proc = normalize(scores_proc)
        
        tokens_filtered, scores_filtered = filter_specials(tokens_raw, scores_proc.tolist(), specials)
        
    except Exception as e:
        print(f"Attribution failed: {e}")
        tokens_raw = tokens_raw or []
        scores_raw = np.zeros(len(tokens_raw))
        scores_proc = np.zeros(len(tokens_raw))
        tokens_filtered, scores_filtered = filter_specials(tokens_raw, scores_proc.tolist(), specials)

    return tokens_filtered, scores_filtered, scores_raw.tolist(), tokens_raw

get_ig_score._needs_grad = True
