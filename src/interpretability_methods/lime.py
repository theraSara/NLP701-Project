import numpy as np 

import torch
from torch.nn.functional import softmax

from utils import filter_specials, special_token_set, normalize

from lime.lime_text import LimeTextExplainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    random_state=42
):
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


