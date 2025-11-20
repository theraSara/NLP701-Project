import os, json
import csv, argparse, random
import numpy as np
import pandas as pd
from copy import deepcopy

import shap
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import load_config, visualize_attributions_html
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.stability import StabilityEvaluator
from evaluation.utils import plot_faithfulness_summary, plot_stability_summary

class SHAPExplainer:
    def __init__(self, model, tokenizer, device, max_evals=100, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_evals = max_evals
        self.max_length = max_length
        self.model.eval()

        def predict_fn(texts):
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
            return probs.detach().cpu().numpy()

        # Text masker: SHAP understands how to mask tokens via the tokenizer
        try:
            text_masker = shap.maskers.Text(self.tokenizer)
        except Exception:
            # Older SHAP 
            text_masker = self.tokenizer

        self.explainer = shap.Explainer(predict_fn, text_masker)

    @torch.no_grad()
    def predict_prob_and_label(self, text):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        probs = torch.softmax(self.model(**enc).logits, dim=-1)
        label = int(torch.argmax(probs, dim=-1).item())
        return float(probs[0, label].item()), label

    def explain_one(self, text):
        """
        Returns IG-compatible explanation dict:
        {
            'tokens','attributions','scores','pred_label','pred_score','confidence'
        }
        """
        sv = self.explainer([text], max_evals=self.max_evals)
        pred_p, pred_y = self.predict_prob_and_label(text)

        tokens = list(sv.data[0])
        values = sv.values[0]
        if isinstance(values, np.ndarray) and values.ndim == 2:
            scores = values[:, pred_y]
        else:
            scores = values
        scores = np.asarray(scores).tolist()

        return {
            "tokens": tokens,
            "attributions": scores,
            "scores": scores,
            "prediction": pred_y,
            "pred_label": pred_y,
            "pred_score": pred_p,
            "confidence": pred_p
        }

    def shuffle_text(self, text, ratio, seed=42):
        random.seed(seed)
        toks = text.split()
        k = int(round(ratio * len(toks)))
        if k < 2 or len(toks) <= 1:
            return text
        idxs = list(range(len(toks)))
        pick = random.sample(idxs, min(k, len(idxs)))
        sub = [toks[i] for i in pick]
        random.shuffle(sub)
        for new_w, j in zip(sub, pick):
            toks[j] = new_w
        return " ".join(toks)

def run_one(config):
    num_samples    = config.get("num_samples", 1)
    dataset_name   = config.get("dataset_name")
    random_samples = config.get("random_samples", False)
    output_dir     = config.get("output_dir", "SHAP_outputs")
    output_format  = config.get("output_format", "json")
    max_evals      = config.get("max_evals", 100)
    max_length     = config.get("max_length", 512)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model & tokenizer
    model_path = config["model_checkpoint_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    csv_path = config.get("csv_path")
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"csv_path not found: {csv_path}")
    df = pd.read_csv(csv_path)

    data = list(zip(df["texts"], df["labels"], df["indices"]))
    if random_samples:
        random.seed(0); random.shuffle(data)
    data = data[:num_samples]

    # Explainer
    shap_exp = SHAPExplainer(model, tokenizer, device, max_evals=max_evals, max_length=max_length)

    # Explanations
    results = []
    keep_special = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "[CLS]", "[SEP]", "[PAD]"}

    for i, (text, true_label, sample_id) in enumerate(data, 1):
        ex = shap_exp.explain_one(text)
        ex.update({
            "sample_id": sample_id,
            "text": text,
            "true_label": true_label
        })

        # Optional HTML visualization (same as IG style)
        vis_tokens = [t for t in ex["tokens"] if t not in keep_special]
        vis_attrs  = [a for t, a in zip(ex["tokens"], ex["attributions"]) if t not in keep_special]
        html = visualize_attributions_html(
            vis_tokens, vis_attrs,
            pred_label=str(ex["pred_label"]),
            pred_score=ex["pred_score"],
            true_label=str(true_label),
            sample_id=sample_id
        )
        with open(os.path.join(output_dir, f"sample_{sample_id}_attribution.html"), "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='UTF-8'></head><body>\n")
            f.write(html)
            f.write("</body></html>")

        results.append(ex)
        print(f"[{i}/{len(data)}] sample {sample_id} — pred={ex['pred_label']}, true={true_label}")

    # Save explanations
    if output_format == "json":
        out_path = os.path.join(output_dir, "shap_explanations.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved SHAP explanations to {out_path}")
    else:
        out_path = os.path.join(output_dir, "shap_explanations.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "token", "attribution", "pred_label", "true_label", "pred_score"])
            for r in results:
                for t, a in zip(r["tokens"], r["attributions"]):
                    w.writerow([r["sample_id"], t, a, r["pred_label"], r["true_label"], r["pred_score"]])
        print(f"Saved SHAP explanations to {out_path}")

    # Evaluators (same wiring as IG)
    eval_cfg = dict(config.get("evaluation_config", {}))

    def predict_prob_for_label(text, label_id):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
            return float(probs[0, label_id].item())

    eval_cfg["tokenizer"] = tokenizer
    eval_cfg["predict_fn"] = predict_prob_for_label

    # Faithfulness
    if config.get("run_faithfulness", False):
        faith_eval = FaithfulnessEvaluator("SHAP", dataset_name, eval_cfg)
        faith = faith_eval.evaluate(results)
        with open(os.path.join(output_dir, "faithfulness_scores.json"), "w") as f:
            json.dump(faith, f, indent=2)
        print("Faithfulness evaluation completed.")
        plot_faithfulness_summary(
            faith,
            os.path.join(output_dir, "faithfulness_curve.png"),
            title=f"{dataset_name} · SHAP (Comprehensiveness & Sufficiency)"
        )

    # Stability
    if config.get("run_stability", False):
        perturbed_path = config.get("stability_perturbed_file")
        generate_perturbed = bool(config.get("generate_perturbed", False))
        ratios = eval_cfg.get("ratios", [0.01, 0.05, 0.10, 0.20, 0.50])

        if not perturbed_path and not generate_perturbed:
            print("No 'stability_perturbed_file' and 'generate_perturbed' is False — skipping stability.")
        else:
            if generate_perturbed:
                # Build SHAP-specific perturbed explanations on-the-fly
                by_id = {}
                for ex in results:
                    sid = str(ex["sample_id"]); txt = ex["text"]
                    by_id[sid] = {}
                    for r in ratios:
                        sh = shap_exp.shuffle_text(txt, r, seed=42)
                        per = shap_exp.explain_one(sh)
                        by_id[sid][f"{r:.2f}"] = {
                            "text": sh,
                            "attributions": per["attributions"],
                            "pred_label": per["pred_label"]
                        }
                perturbed_explanations = {"by_id": by_id}

                outp = os.path.join(output_dir, "perturbed_shap.json")
                with open(outp, "w", encoding="utf-8") as f:
                    json.dump(perturbed_explanations, f, indent=2)
                print(f"Saved SHAP perturbed explanations to {outp}")
            else:
                if not os.path.exists(perturbed_path):
                    print(f"[warn] Provided 'stability_perturbed_file' not found: {perturbed_path} — skipping stability.")
                    perturbed_explanations = None
                else:
                    with open(perturbed_path, "r", encoding="utf-8") as f:
                        perturbed_explanations = json.load(f)

            if perturbed_explanations:
                stab_eval = StabilityEvaluator("SHAP", dataset_name, eval_cfg)
                stab = stab_eval.evaluate(results, perturbed_explanations)
                with open(os.path.join(output_dir, "stability_scores.json"), "w") as f:
                    json.dump(stab, f, indent=2)
                print("Stability evaluation completed.")
                plot_stability_summary(
                    stab,
                    os.path.join(output_dir, "stability_curve.png"),
                    title=f"{dataset_name} · SHAP ({eval_cfg.get('similarity_metric','spearman').capitalize()})"
                )
    print(f"Done. Outputs in: {output_dir}")

def merge_common(common: dict, task: dict) -> dict:
    cfg = deepcopy(common)
    cfg.update(task)
    return cfg

def run_with_config_or_batch(path):
    cfg = load_config(path)
    if isinstance(cfg, dict) and "tasks" in cfg:
        common = cfg.get("common", {})
        tasks  = cfg["tasks"]
        for i, t in enumerate(tasks, 1):
            merged = merge_common(common, t)
            print(f"[{i}/{len(tasks)}] SHAP — model={merged.get('model_checkpoint_path')}  dataset={merged.get('dataset_name')}")
            run_one(merged)
    else:
        run_one(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to config JSON (single task or batch)")
    args = ap.parse_args()
    run_with_config_or_batch(args.config)

if __name__ == "__main__":
    main()
