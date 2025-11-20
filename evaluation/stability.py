import numpy as np
from evaluation.base import BaseEvaluator

try:
    from scipy.stats import spearmanr, pearsonr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

class StabilityEvaluator(BaseEvaluator):
    def __init__(self, method_name, dataset_name, config):
        super().__init__(method_name, dataset_name, config)
        self.similarity_metric = (config.get("similarity_metric") or "spearman").lower()
        self.ratios = config.get("ratios")

    def similarity(self, v1, v2):
        a = np.asarray(v1, dtype=float); b = np.asarray(v2, dtype=float)
        L = min(len(a), len(b))
        if L < 2: return np.nan
        a = a[:L]; b = b[:L]
        if self.similarity_metric == "spearman":
            if _HAS_SCIPY:
                r, _ = spearmanr(a, b); return float(r)
            ra = a.argsort().argsort().astype(float)
            rb = b.argsort().argsort().astype(float)
            return float(np.corrcoef(ra, rb)[0,1])
        if self.similarity_metric == "pearson":
            if _HAS_SCIPY:
                r, _ = pearsonr(a, b); return float(r)
            return float(np.corrcoef(a, b)[0,1])
        if self.similarity_metric == "cosine":
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0: return np.nan
            return float(np.dot(a, b) / (na * nb))
        if self.similarity_metric == "l2":
            return -float(np.linalg.norm(a - b))
        raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    def curve_auc(self, curve):
        if not curve: return float("nan"), float("nan")
        xs, ys = [], []
        for k, v in curve.items():
            try:
                xs.append(float(k)); ys.append(float(v))
            except Exception:
                continue
        if len(xs) < 2: return float("nan"), float("nan")
        idx = np.argsort(xs); xs = np.asarray(xs)[idx]; ys = np.asarray(ys)[idx]
        auc = float(np.trapz(ys, xs))
        span = float(xs[-1] - xs[0])
        auc_norm = float(auc / span) if span > 0 else float("nan")
        return auc, auc_norm

    def coerce(self, pertd):
        if isinstance(pertd, dict) and "by_id" in pertd:
            return pertd["by_id"]
        return pertd

    def evaluate(self, original_explanations, perturbed_explanations):
        pertd = self.coerce(perturbed_explanations)

        # original by id
        orig_by_id = {}
        for ex in original_explanations:
            sid = str(ex.get("sample_id", ex.get("id")))
            orig_by_id[sid] = ex

        # find ratios
        all_ratios = set()
        for sid, mapping in pertd.items():
            for r in mapping.keys():
                try:
                    all_ratios.add("{:.2f}".format(float(r)))
                except Exception:
                    pass
        if not all_ratios:
            return {"per_sample": [], "aggregate": {"by_ratio": {}}}
        ordered = ["{:.2f}".format(float(r)) for r in (self.ratios or sorted(list(all_ratios), key=lambda x: float(x)))]

        per_sample = []
        bucket = {r: [] for r in ordered}
        auc_vals, aucn_vals = [], []

        for sid, oex in orig_by_id.items():
            if sid not in pertd: continue
            oimp = oex.get("attributions") or oex.get("importances")
            if oimp is None: continue

            curve = {}
            for r in ordered:
                pex = pertd[sid].get(r)
                if not pex: continue
                pimp = pex.get("attributions") or pex.get("importances")
                if pimp is None: continue
                val = self.similarity(oimp, pimp)
                if not np.isnan(val):
                    curve[r] = val
                    bucket[r].append(val)

            auc, aucn = self.curve_auc(curve)
            if not np.isnan(auc):  auc_vals.append(auc)
            if not np.isnan(aucn): aucn_vals.append(aucn)

            per_sample.append({"id": sid, "by_ratio": curve, "auc": auc, "auc_norm": aucn})

        def summ(vals):
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": int(len(vals))} if vals else {"mean": float("nan"), "std": float("nan"), "n": 0}

        by_ratio = {}
        for r in ordered:
            by_ratio[r] = summ(bucket.get(r, []))

        return {
            "per_sample": per_sample,
            "aggregate": {
                "by_ratio": by_ratio,
                "auc": summ(auc_vals),
                "auc_norm": summ(aucn_vals)
            }
        }
