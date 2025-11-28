import torch

from utils import filter_specials, special_token_set, normalize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === ATTENTION METHOD ===
@torch.no_grad()
def get_attention_score(
    text,
    tokenizer,
    model,
    device =device,
    max_length= 256,
):
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
