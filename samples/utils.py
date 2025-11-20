import random 
import numpy as np
import pickle 
import pandas as pd 
import matplotlib.pyplot as plt
import os

def set_seed(SEED=42): 
    random.seed(SEED)
    np.random.seed(SEED)

def load_data(data_path): 
    with open(data_path, 'rb') as f:
        samples = pickle.load(f)
        print(f"Loaded samples from: {data_path}")
        df = pd.DataFrame(samples)
        return df 

def special_token_set(tokenizer):
    return {
        getattr(tokenizer, "cls_token", None),
        getattr(tokenizer, "sep_token", None),
        getattr(tokenizer, "pad_token", None),
        getattr(tokenizer, "bos_token", None),
        getattr(tokenizer, "eos_token", None),
    }

def filter_specials(tokens, scores, SPECIALS):
    keep = [i for i, t in enumerate(tokens) if t not in SPECIALS]
    if not keep:
        # nothing left — return raw (or return empty & handle upstream)
        return tokens, scores
    return [tokens[i] for i in keep], [scores[i] for i in keep] 
    
def plot_stability(combined_df, RATIOS, title):
    df = combined_df.copy()
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["p"]).sort_values(["model", "p"])

    plt.figure(figsize=(6, 4))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["p"] * 100, sub["corr_spearman_mean"], marker="o", label=model)

    plt.title(title)
    plt.xlabel("Shuffle rate (%)")
    plt.ylabel("Stability")
    plt.legend()
    
    # --- use RATIOS for x-ticks ---
    plt.xticks([r * 100 for r in RATIOS], [int(r * 100) for r in RATIOS])
    plt.ylim()
    plt.tight_layout()
    
    save_path = f"eval_plot/{title}_stability.png"
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    plt.show()

def plot_comprehensiveness(combined_df, RATIOS, title):
    """
    Plot Comprehensiveness (confidence drop) for multiple models.
    Expected columns: ['model', 'p', 'comp_mean']
    """
    df = combined_df.copy()
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["p"]).sort_values(["model", "p"])

    plt.figure(figsize=(6, 4))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["p"] * 100, sub["comp_mean"], marker="o", label=model)

    plt.title(f"{title}")
    plt.xlabel("Removed ratio (%)")
    plt.ylabel("Comprehensiveness")
    plt.legend()

    # --- use RATIOS for x-ticks ---
    plt.xticks([r * 100 for r in RATIOS], [int(r * 100) for r in RATIOS])
    plt.tight_layout()
    
    save_path = f"eval_plot/{title}_comprehensiveness.png"
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    plt.show()

def plot_sufficiency(combined_df, RATIOS, title):
    """
    Plot Sufficiency (confidence retention) for multiple models.
    Expected columns: ['model', 'p', 'suff_mean']
    """
    df = combined_df.copy()
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["p"]).sort_values(["model", "p"])

    plt.figure(figsize=(6, 4))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["p"] * 100, sub["suff_mean"], marker="o", label=model)

    plt.title(f"{title}")
    plt.xlabel("Kept ratio (%)")
    plt.ylabel("Sufficiency")
    plt.legend()

    # --- use RATIOS for x-ticks ---
    plt.xticks([r * 100 for r in RATIOS], [int(r * 100) for r in RATIOS])
    plt.tight_layout()
    
    save_path = f"eval_plot/{title}_sufficiency.png"
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    plt.show()