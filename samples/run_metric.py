import sys
from pathlib import Path
from typing import Callable, Dict

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import set_seed, load_data, plot_stability, plot_comprehensiveness, plot_sufficiency
from token_score import get_attention_score, get_lime_score, load_stability_json, load_faithfulness_json
from eval_stability import stability
from eval_faithfulness import comprehensiveness, sufficiency

# Make DEVICE available to all functions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------- Utilities ----------------------
def ensure_dir(path: str | Path) -> Path:  # if <3.10: use Union[str, Path]
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, output_attentions=True
    )
    model.to(DEVICE).eval()
    return tokenizer, model

def run_metric_over_all(
    *,
    metric_name: str,
    run_fn: Callable,
    get_importance_fn: Callable,
    method_label: str,
    out_dir: Path,
):
    """
    Runs one metric (stability/comprehensiveness/sufficiency) over all datasets/models.
    Saves a CSV per dataset.
    """
    for dataset_name, cfg in datasets.items():
        print(f"\n=== Processing {dataset_name} :: {metric_name} :: {method_label} ===")
        all_results = []

        for model_name, model_dir in cfg["models"].items():
            print(f"\n----- Running {metric_name} for {model_name} ({dataset_name}) -----")
            tokenizer, model = load_model_and_tokenizer(model_dir)

            with torch.no_grad():
                df = run_fn(
                    cfg["data"],  # already a DataFrame
                    tokenizer,
                    model,
                    get_importance_fn,
                    RATIOS,
                    DEVICE,
                )

            # Standardize columns
            if metric_name == "stability":
                df.rename(columns={"ratio": "p", "mean_spearman": "corr_spearman_mean"}, inplace=True)
            else:
                df.rename(columns={"ratio": "p"}, inplace=True)

            df["model"] = model_name
            all_results.append(df)

        final_df = pd.concat(all_results, ignore_index=True)

        # Save & plot
        out_dir = ensure_dir(out_dir)  # just in case
        csv_path = out_dir / f"{dataset_name}_{metric_name}_{method_label}.csv"
        final_df.to_csv(csv_path, index=False)
        print(final_df)

        if metric_name == "stability":
            plot_stability(final_df, RATIOS, out_dir, method_label, dataset_name)
        elif metric_name == "comprehensiveness":
            plot_comprehensiveness(final_df, RATIOS, out_dir, method_label, dataset_name)
        elif metric_name == "sufficiency":
            plot_sufficiency(final_df, RATIOS, out_dir, method_label, dataset_name)

def main(selected_methods: list[str] | None = None, output_dir: str = "results"):
    out_dir = ensure_dir(output_dir)

    methods_to_run = selected_methods or list(METHODS.keys())
    for method_key in methods_to_run:
        if method_key not in METHODS:
            print(f"[WARN] Unknown method '{method_key}' — skipping.")
            continue

        method_label = METHODS[method_key]["label"]
        get_importance_fn = METHODS[method_key]["fn"]

        # 1) Stability
        run_metric_over_all(
            metric_name="stability",
            run_fn=stability,
            get_importance_fn=get_importance_fn,
            method_label=method_label,
            out_dir=out_dir,
        )

        # 2) Comprehensiveness
        run_metric_over_all(
            metric_name="comprehensiveness",
            run_fn=comprehensiveness,
            get_importance_fn=get_importance_fn,
            method_label=method_label,
            out_dir=out_dir,
        )

        # 3) Sufficiency
        run_metric_over_all(
            metric_name="sufficiency",
            run_fn=sufficiency,
            get_importance_fn=get_importance_fn,
            method_label=method_label,
            out_dir=out_dir,
        )

if __name__ == "__main__":
    set_seed()

    # imdb_path = 'sampled/imdb_sampled_500.pkl'
    # sst2_path = 'sampled/sst2_sampled_436.pkl'

    # n_samples = 1
    # imdb_df = load_data(imdb_path).iloc[:n_samples]
    # sst2_df = load_data(sst2_path).iloc[:n_samples]

    RATIOS = [0.01, 0.05, 0.10, 0.20, 0.50]

    # datasets: Dict[str, Dict] = {
    #     "SST2": {
    #         "data": sst2_df,
    #         "models": {
    #             "TinyBERT": r"D:/master/NLP/models/tinybert_sst2/final",
    #             "DistilBERT": r"D:/master/NLP/models/distilbert_sst2/final",
    #             "ALBERT": r"D:/master/NLP/models/albert_sst2/final",
    #         },
    #     },
    #     "IMDB": {
    #         "data": imdb_df,
    #         "models": {
    #             "TinyBERT": r"D:/master/NLP/models/tinybert_imdb/final",
    #             "DistilBERT": r"D:/master/NLP/models/distilbert_imdb/final",
    #             "ALBERT": r"D:/master/NLP/models/albert_imdb/final",
    #         },
    #     },
    # }
    
    dataset_name = "SST"
    out_dir = "results"

    if dataset_name == "SST2":
        runs = {
            "DistilBERT": "ig_outputs/sst2_distilbert",
            "TinyBERT":   "ig_outputs/sst2_tinybert",
            "ALBERT":     "ig_outputs/sst2_albert",
        }
    else:
        runs = {
            "DistilBERT": "ig_outputs/imdb_distilbert",
            "TinyBERT":   "ig_outputs/imdb_tinybert",
            "ALBERT":     "ig_outputs/imdb_albert",
        }
        
    comp_dfs, suff_dfs, stab_dfs = [], [], []
    for model_name, base in runs.items():
        comp_df, suff_df = load_faithfulness_json(
            f"{base}/faithfulness_scores.json", model_name
        )
        stab_df = load_stability_json(
            f"{base}/stability_scores.json", model_name
        )
        comp_dfs.append(comp_df)
        suff_dfs.append(suff_df)
        stab_dfs.append(stab_df)

    comp_all = pd.concat(comp_dfs, ignore_index=True)
    suff_all = pd.concat(suff_dfs, ignore_index=True)
    stab_all = pd.concat(stab_dfs, ignore_index=True)

    plot_comprehensiveness(
        comp_all, RATIOS, out_dir,
        method_label="Integrated Gradients",
        dataset_name=dataset_name
    )

    plot_sufficiency(
        suff_all, RATIOS, out_dir,
        method_label="Integrated Gradients",
        dataset_name=dataset_name
    )

    plot_stability(
        stab_all, RATIOS, out_dir,
        method_label="Integrated Gradients",
        dataset_name=dataset_name
    )

    # METHODS: Dict[str, Dict[str, Callable]] = {
    #     "attention": {"label": "Attention", "fn": get_attention_score},
    #     "lime": {"label": "LIME", "fn": get_lime_score},
    #     # "ig": {"label": "Grad", "fn": get_grad_score},
    # }

    #methods = sys.argv[1:] if len(sys.argv) > 1 else None
    #main(methods)
