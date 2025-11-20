
import os
import matplotlib.pyplot as plt


def plot_faithfulness_summary(faith_dict, output_path, title=None):
    """
    FaithfulnessEvaluator output -> two lines:
      - Comprehensiveness mean vs ratio
      - Sufficiency mean vs ratio
    """
    agg = faith_dict.get("aggregate", {})
    comp = agg.get("comprehensiveness", {})
    suff = agg.get("sufficiency", {})

    # collect ratios (as floats) that have data
    ratios = []
    comp_means = []
    suff_means = []

    keys = sorted(comp.keys(), key=lambda k: float(k))
    for k in keys:
        c = comp.get(k, {})
        s = suff.get(k, {})
        if c.get("n", 0) > 0 and s.get("n", 0) > 0:
            ratios.append(float(k))
            comp_means.append(float(c.get("mean", 0.0)))
            suff_means.append(float(s.get("mean", 0.0)))

    if not ratios:
        print("plot_faithfulness_summary: no data to plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot([r * 100 for r in ratios], comp_means, marker="o", label="Comprehensiveness")
    plt.plot([r * 100 for r in ratios], suff_means, marker="o", label="Sufficiency")
    plt.xlabel("Ratio (%)")
    plt.ylabel("Score")
    if title:
        plt.title(title)
    plt.xticks([int(r * 100) for r in ratios], [int(r * 100) for r in ratios])
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_stability_summary(stab_dict, output_path, title=None):
    """
    StabilityEvaluator output -> one line: mean correlation vs shuffle ratio.
    Expects stab_dict["aggregate"]["by_ratio"][ratio_str] = {"mean": ..., "n": ...}
    """
    agg = stab_dict.get("aggregate", {})
    by_ratio = agg.get("by_ratio", {})

    ratios = []
    means = []

    keys = sorted(by_ratio.keys(), key=lambda k: float(k))
    for k in keys:
        s = by_ratio[k]
        if s.get("n", 0) > 0:
            ratios.append(float(k))
            means.append(float(s.get("mean", 0.0)))

    if not ratios:
        print("plot_stability_summary: no data to plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot([r * 100 for r in ratios], means, marker="o")
    plt.xlabel("Shuffle rate (%)")
    plt.ylabel("Stability (corr)")
    if title:
        plt.title(title)
    plt.xticks([int(r * 100) for r in ratios], [int(r * 100) for r in ratios])
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
