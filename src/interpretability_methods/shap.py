import numpy as np 

import torch

from utils import filter_specials, special_token_set, normalize

import shap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## === SHAP METHOD ===

# simple alignment of SHAP word-level scores to model token-level scores
def align_shap_simple(shap_tokens, shap_scores, model_tokens, specials):
    scores = np.zeros(len(model_tokens))
    specials = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}

    # Clean tokens for rough matching
    def clean(t):
        return t.replace('##', '').replace('▁', '').replace('Ġ', '').strip().lower()
    
    shap_words = [clean(t) for t in shap_tokens]
    model_words = [clean(t) for t in model_tokens]
    
    shap_idx = 0
    for i, model_word in enumerate(model_words):
        # Skip special tokens
        if model_tokens[i] in specials:
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
    attention_mask = inputs['attention_mask']
    seq_len = len(input_ids)
    
    # Baseline (logit for the predicted class)
    with torch.no_grad():
        baseline_logits = model(**inputs).logits[0, pred_class].item()
    
    # Mask each token
    scores = np.zeros(seq_len)
    # use tokenizer's mask token ID, falling back to PAD ID or 0
    mask_id = getattr(tokenizer, 'mask_token_id', tokenizer.pad_token_id or 0)
    
    for i in range(seq_len):
        masked_ids = input_ids.clone()
        # only mask if it's not a special token itself
        tokens_raw = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        if tokens_raw[i] not in special_token_set(tokenizer):
            masked_ids[i] = mask_id
        
        with torch.no_grad():
            masked_output = model(
                input_ids=masked_ids.unsqueeze(0),
                attention_mask=attention_mask
            )
            masked_logits = masked_output.logits[0, pred_class].item()
        
        scores[i] = baseline_logits - masked_logits
    
    return scores

# Computes SHAP attribution scores using the Explainer class
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
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        pred_class = outputs.logits.argmax(dim=-1).item()

    # get the set of special tokens 
    specials = special_token_set(tokenizer)
    
    try:
        # Create prediction function
        def predict_fn(texts):
            # SHAP may pass str, list of str, or np.ndarray
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, np.ndarray):
                texts = texts.flatten().tolist()
            elif isinstance(texts, list):
                # flatten list of lists from SHAP text splitting if needed 
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
            # SHAP expects output probabilities for classes
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
        scores_raw = align_shap_simple(shap_tokens, shap_scores, tokens, specials)
        
    except Exception as e:
        print(f"SHAP failed on text (len={len(text)}): {str(e)[:100]}")
        print(f"Falling back to masking-based importance")
        
        # masking-based importance
        scores_raw = compute_masking_importance(
            text, tokenizer, model, device, max_length, pred_class
        )
    
    # Process scores
    scores_processed = np.abs(scores_raw) if use_abs else np.array(scores_raw).copy()
    scores_raw = np.array(scores_raw) 
    
    if l1_normalize:
        scores_processed = normalize(scores_processed)
        
    tokens_filtered, scores_filtered = filter_specials(tokens, scores_processed.tolist(), specials)

    return tokens_filtered, scores_filtered, scores_raw.tolist(), tokens