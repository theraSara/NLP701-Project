import os, sys
import json, csv
import random
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

from .utils import load_config, visualize_attributions_html
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.stability import StabilityEvaluator
from evaluation.utils import plot_faithfulness_summary, plot_stability_summary

def run_with_config(config):
    num_samples = config.get("num_samples", 10)
    # model_name = config.get("model_name") or config.get("model_path")
    dataset_name = config.get("dataset_name")
    # dataset_config = config.get("dataset_config", None)
    # dataset_split = config.get("dataset_split", "validation")
    random_samples = config.get("random_samples", False)
    output_dir = config.get("output_dir", "IG_outputs")
    output_format = config.get("output_format", "json")
    ig_steps = config.get("ig_steps", 50)
    internal_bs = config.get("internal_batch_size", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # load model and tokenizer
    model_path = config["model_checkpoint_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    model.zero_grad()

    # read CSV
    csv_path = config.get("csv_path")
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"csv_path not found: {csv_path}")
    df = pd.read_csv(csv_path)
    data = list(zip(df["texts"], df["labels"], df["indices"]))

    if random_samples:
        random.seed(0)
        random.shuffle(data)
    data = data[:num_samples]

    # get embedding layer for IG
    if hasattr(model, 'bert'):
        embed_layer = model.bert.embeddings
    elif hasattr(model, 'distilbert'):
        embed_layer = model.distilbert.embeddings
    elif hasattr(model, 'albert'):
        embed_layer = model.albert.embeddings
    else:
        embed_layer = model.get_input_embeddings()
        if embed_layer is None:
            raise RuntimeError("Couldn't find embeddings layer in the model.")

    # detect whether token_type_ids are needed
    use_token_type = 'token_type_ids' in AutoTokenizer.from_pretrained(model_path)("test", return_tensors='pt')

    # forward
    if use_token_type:
        def forward_func(input_ids, attention_mask, token_type_ids):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            return outputs.logits
    else:
        def forward_func(input_ids, attention_mask):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    lig = LayerIntegratedGradients(forward_func, embed_layer)

    results = []
    for count, (text, true_label, sample_id) in enumerate(data, 1):
        enc = AutoTokenizer.from_pretrained(model_path)(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        input_ids_tensor = enc['input_ids']
        attention_mask_tensor = enc['attention_mask']
        token_type_ids_tensor = enc.get('token_type_ids', None)

        # model prediction
        with torch.no_grad():
            logits = model(**enc).logits
            target_idx = int(torch.argmax(logits, dim=-1).item())
            probs = torch.softmax(logits, dim=-1)
            pred_score = float(probs[0, target_idx].item())

        # baseline: pad everywhere except specials
        input_ids_list = input_ids_tensor[0].tolist()
        special_ids = {
            t for t in [
                tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id,
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
            ] if t is not None
        }
        baseline_ids = [tid if tid in special_ids else tokenizer.pad_token_id for tid in input_ids_list]
        baseline_tensor = torch.tensor([baseline_ids], device=device)

        # IG Attribution
        if use_token_type and token_type_ids_tensor is not None:
            attributions, delta = lig.attribute(
                inputs=input_ids_tensor,
                baselines=baseline_tensor,
                additional_forward_args=(attention_mask_tensor, token_type_ids_tensor),
                target=target_idx,
                n_steps=ig_steps,
                internal_batch_size=internal_bs,
                return_convergence_delta=True
            )
        else:
            attributions, delta = lig.attribute(
                inputs=input_ids_tensor,
                baselines=baseline_tensor,
                additional_forward_args=(attention_mask_tensor,),
                target=target_idx,
                n_steps=ig_steps,
                internal_batch_size=internal_bs,
                return_convergence_delta=True
            )

        token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids_list)

        # HTML viz
        viz_keep = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, '[CLS]', '[SEP]', '[PAD]'}
        html_content = visualize_attributions_html(
            [t for t in tokens if t not in viz_keep],
            [a for t, a in zip(tokens, token_scores) if t not in viz_keep],
            pred_label=str(target_idx),
            pred_score=pred_score,
            true_label=str(true_label),
            sample_id=sample_id
        )

        html_file = os.path.join(output_dir, f"sample_{sample_id}_attribution.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write("<html><head><meta charset='UTF-8'></head><body>\n")
            f.write(html_content)
            f.write("</body></html>")

        results.append({
            "sample_id": sample_id,
            "text": text,
            "tokens": tokens,
            "attributions": token_scores,
            "pred_label": target_idx,
            "true_label": true_label,
            "pred_score": pred_score,
            "delta": float(delta.item()) if isinstance(delta, torch.Tensor) else float(delta)
        })

        print(f"[{count}/{len(data)}] Processed sample {sample_id} — Pred: {target_idx}, True: {true_label}")

    # save attributions
    if output_format == "json":
        out_path = os.path.join(output_dir, "attributions.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Attribution scores saved to {out_path}")
    elif output_format == "csv":
        out_path = os.path.join(output_dir, "attributions.csv")
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "token", "attribution", "pred_label", "true_label", "pred_score", "delta"])
            for entry in results:
                for tok, attr in zip(entry["tokens"], entry["attributions"]):
                    writer.writerow([entry["sample_id"], tok, attr, entry["pred_label"], entry["true_label"], entry["pred_score"], entry["delta"]])
        print(f"Attribution scores saved to {out_path}")

    # evaluation
    eval_config = config.get("evaluation_config", {})

    def predict_prob_for_label(text, label_id):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
            return float(probs[0, label_id].item())

    eval_cfg = dict(eval_config)
    eval_cfg["tokenizer"] = tokenizer
    eval_cfg["predict_fn"] = predict_prob_for_label

    if config.get("run_faithfulness", False):
        evaluator = FaithfulnessEvaluator("IG", dataset_name, eval_cfg)
        faith = evaluator.evaluate(results)
        with open(os.path.join(output_dir, "faithfulness_scores.json"), "w") as f:
            json.dump(faith, f, indent=2)
        print("Faithfulness evaluation completed.")
        plot_faithfulness_summary(
            faith,
            os.path.join(output_dir, "faithfulness_curve.png"),
            title=f"{dataset_name} · IG (Comprehensiveness & Sufficiency)"
        )

    if config.get("run_stability", False):
        perturbed_path = config.get("stability_perturbed_file")
        if not perturbed_path or not os.path.exists(perturbed_path):
            print(f"[warn] stability_perturbed_file missing: {perturbed_path}. Skipping stability for this run.")
        else:
            with open(perturbed_path, 'r', encoding='utf-8') as f:
                perturbed_explanations = json.load(f)
            eval_config_for_stability = dict(eval_config)
            eval_config_for_stability["dataset_name"] = dataset_name
            stab_eval = StabilityEvaluator("IG", dataset_name, eval_config_for_stability)
            stab = stab_eval.evaluate(results, perturbed_explanations)
            with open(os.path.join(output_dir, "stability_scores.json"), "w") as f:
                json.dump(stab, f, indent=2)
            print("Stability evaluation completed.")
            plot_stability_summary(
                stab,
                os.path.join(output_dir, "stability_curve.png"),
                title=f"{dataset_name} · IG ({eval_config.get('similarity_metric','spearman').capitalize()})"
            )

def main(config_path):
    config = load_config(config_path)
    common_cfg = config.get("common", {})
    tasks = config.get("tasks", [])

    for task_cfg in tasks:
        # merge common settings into each task
        merged_cfg = {**common_cfg, **task_cfg}
        run_with_config(merged_cfg)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    main(sys.argv[1])
