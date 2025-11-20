import os
import json
import argparse
import pandas as pd

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

def _parse_ratio(colname):
    # shuf_001 -> 0.01, shuf_010 -> 0.10, shuf_1.00 -> 1.00, shuf_100 -> 1.00
    if not colname.startswith("shuf_"):
        return None
    token = colname.split("shuf_")[-1].strip().replace("%", "")
    try:
        if "." in token:
            val = float(token)
            if val > 1.0:  # e.g., 50.0 -> 0.50
                val = val / 100.0
        else:
            val = int(token) / 100.0
    except Exception:
        try:
            val = float(token) / 100.0
        except Exception:
            return None
    return "{:.2f}".format(val)

def _get_embed_layer(model):
    if hasattr(model, "bert"):
        return model.bert.embeddings
    if hasattr(model, "distilbert"):
        return model.distilbert.embeddings
    if hasattr(model, "albert"):
        return model.albert.embeddings
    emb = model.get_input_embeddings()
    if emb is None:
        raise RuntimeError("Could not find embedding layer.")
    return emb

def _make_forward_fn(model, tokenizer):
    need_tti = "token_type_ids" in tokenizer("test", return_tensors="pt")
    if need_tti:
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
    return forward_func, need_tti

def build(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device).eval().zero_grad()

    embed_layer = _get_embed_layer(model)
    forward_func, need_tti = _make_forward_fn(model, tokenizer)
    lig = LayerIntegratedGradients(forward_func, embed_layer)

    df = pd.read_csv(args.csv_path)

    # detect shuffle columns
    shuf_cols, col2ratio = [], {}
    for c in df.columns:
        r = _parse_ratio(c)
        if r is not None:
            shuf_cols.append(c)
            col2ratio[c] = r
    if not shuf_cols:
        raise ValueError("No 'shuf_XXX' columns found in CSV.")

    by_id = {}
    for _, row in df.iterrows():
        sample_id = row["indices"]
        by_id[str(sample_id)] = {}
        for col in shuf_cols:
            text_p = str(row[col])
            enc = tokenizer(text_p, return_tensors="pt", truncation=True, max_length=256).to(device)
            input_ids = enc["input_ids"]; attn_mask = enc["attention_mask"]
            token_type_ids = enc.get("token_type_ids", None)

            with torch.no_grad():
                logits = model(**enc).logits
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                probs = torch.softmax(logits, dim=-1)
                pred_score = float(probs[0, pred_idx].item())

            input_ids_list = input_ids[0].tolist()
            special_ids = {
                t for t in [
                    tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id,
                    getattr(tokenizer, "bos_token_id", None),
                    getattr(tokenizer, "eos_token_id", None),
                ] if t is not None
            }
            baseline_ids = [tid if tid in special_ids else tokenizer.pad_token_id for tid in input_ids_list]
            baseline = torch.tensor([baseline_ids], device=device)

            if need_tti and token_type_ids is not None:
                atts = lig.attribute(
                    inputs=input_ids,
                    baselines=baseline,
                    additional_forward_args=(attn_mask, token_type_ids),
                    target=pred_idx,
                    n_steps=args.ig_steps,
                    internal_batch_size=args.internal_bs,
                    return_convergence_delta=False
                )
                
            else:
                atts = lig.attribute(
                    inputs=input_ids,
                    baselines=baseline,
                    additional_forward_args=(attn_mask,),
                    target=pred_idx,
                    n_steps=args.ig_steps,
                    internal_batch_size=args.internal_bs,
                    return_convergence_delta=False
                )

            token_scores = atts.sum(dim=-1).squeeze(0).detach().cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids_list)

            ratio_str = col2ratio[col]
            by_id[str(sample_id)][ratio_str] = {
                "sample_id": sample_id,
                "ratio": ratio_str,
                "text": text_p,
                "tokens": tokens,
                "attributions": token_scores,
                "pred_label": pred_idx,
                "pred_score": pred_score
            }

    out_path = os.path.join(args.output_dir, "perturbed_ig_imdb.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"by_id": by_id}, f, indent=2)
    print(f"Saved perturbed explanations to: {out_path}")
    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ig_steps", type=int, default=16)
    ap.add_argument("--internal_bs", type=int, default=None)
    args = ap.parse_args()
    build(args)


"""
python src/evaluation/perturbed_explanations.py \
  --model_path ./models/albert_sst2/final \
  --csv_path   ./data/permutated100/sst2_sample_permutated.csv \
  --output_dir ig_outputs/sst2_distilbert_ig

python src/evaluation/perturbed_explanations.py \
  --model_path ./models/distilbert_sst2/final \
  --csv_path   ./data/permutated100/sst2_sample_permutated.csv \
  --output_dir ig_outputs/distilbert_ig_sst2

python src/evaluation/perturbed_explanations.py \
  --model_path ./models/tinybert_sst2/final \
  --csv_path   ./data/permutated100/sst2_sample_permutated.csv \
  --output_dir ig_outputs/tinybert_ig_sst2

python src/evaluation/perturbed_explanations.py \
  --model_path ./models/albert_imdb/final \
  --csv_path   ./data/permutated100/imdb_sample_permutated.csv \
  --output_dir ig_outputs/imdb_distilbert_ig

python src/evaluation/perturbed_explanations.py \
  --model_path ./models/albert_imdb/final \
  --csv_path   ./data/permutated100/imdb_sample_permutated.csv \
  --output_dir ig_outputs/distilbert_ig_imdb

python src/evaluation/perturbed_explanations.py \
  --model_path ./models/tinybert_imdb/final \
  --csv_path   ./data/permutated100/imdb_sample_permutated.csv \
  --output_dir ig_outputs/tinybert_ig_imdb
"""