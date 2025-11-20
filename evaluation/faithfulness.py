import numpy as np
from evaluation.base import BaseEvaluator

class FaithfulnessEvaluator(BaseEvaluator):
    # Computes ratio-based Comprehensiveness & Sufficiency for each explanation.
    def __init__(self, method_name, dataset_name, config):
        super().__init__(method_name, dataset_name, config)

        self.tokenizer  = config.get("tokenizer")
        self.predict_fn = config.get("predict_fn") 
        if self.tokenizer is None:
            raise ValueError("FaithfulnessEvaluator requires 'tokenizer' in config.")
        if self.predict_fn is None:
            raise ValueError("FaithfulnessEvaluator requires 'predict_fn' in config (text, target_label) -> prob.")

        self.ratios = config.get("ratios", [0.01, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00])
        self.normalize_auc = bool(config.get("normalize_auc", True))

        # Precompute special token id set
        sids = set()
        for name in ["cls_token_id", "sep_token_id", "pad_token_id", "bos_token_id", "eos_token_id"]:
            v = getattr(self.tokenizer, name, None)
            if v is not None:
                sids.add(v)
        self.special_ids = sids

    def encode_ids(self, text):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        return enc["input_ids"][0].tolist()

    def get_importances(self, ex):
        if "importances" in ex:
            return np.asarray(ex["importances"], dtype=float)
        if "attributions" in ex:
            return np.asarray(ex["attributions"], dtype=float)
        raise KeyError("Explanation must include 'importances' or 'attributions'.")
    
    def curve_auc(self, curve_dict):
        if not curve_dict:
            return float("nan"), float("nan")
        xs, ys = [], []
        for k, v in curve_dict.items():
            try:
                x = float(k)
                y = float(v)
            except Exception:
                continue
            xs.append(x)
            ys.append(y)

        if len(xs) < 2:
            return float("nan"), float("nan")
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        auc = float(np.trapz(ys, xs))
        span = float(xs[-1]-xs[0])
        auc_norm = float(auc/span) if span > 0 else float("nan")
        return auc, auc_norm
    
    def comprehensiveness(self, text, tokens, importances, target_label):
        """
        Delete top-q% (by importance) among NON-special positions.
        Decode kept IDs: [CLS] + kept + [SEP], then measure prob drop.
        """
        ids = self.encode_ids(text)
        L = min(len(ids), len(tokens), len(importances))
        if L < 3:
            return {}  

        ids = ids[:L]
        tokens = tokens[:L]
        importances = np.asarray(importances[:L], dtype=float)

        # candidate positions = non-special ids
        cand = [i for i, tid in enumerate(ids) if tid not in self.special_ids]
        if not cand:
            return {}
        
        # decresing order candidates by importance
        scores = np.argsort(importances[cand])
        order = np.array(cand)[scores[::-1]]

        # target class (original prob)
        p_orig = float(self.predict_fn(text, target_label))
        out = {}
        for r in self.ratios:
            k = int(round(r * len(cand)))
            k = max(1, min(k, len(cand) - 1))  # keep at least 1 token in the middle
            remove = set(order[:k])

            kept = [ids[0]] + [ids[i] for i in range(1, L - 1) if i not in remove] + [ids[-1]]
            reduced_text = self.tokenizer.decode(kept, skip_special_tokens=True)
            p_red = float(self.predict_fn(reduced_text, target_label))
            out[r] = p_orig - p_red   # higher is better
        return out
    
    def sufficiency(self, text, tokens, importances, target_label):
        """
        Keep ONLY top-q% (by importance) among NON-special positions.
        Decode kept IDs: [CLS] + top-k + [SEP], then measure prob drop.
        """
        ids = self.encode_ids(text)
        L = min(len(ids), len(tokens), len(importances))
        if L < 3:
            return {}

        ids = ids[:L]
        tokens = tokens[:L]
        importances = importances[:L]

        cand = [i for i, tid in enumerate(ids) if tid not in self.special_ids]
        if not cand:
            return {}
        
        scores = np.argsort(importances[cand])
        order = np.array(cand)[scores[::-1]]

        p_orig = float(self.predict_fn(text, target_label))
        out = {}
        for r in self.ratios:
            k = int(round(r * len(cand)))
            k = max(1, min(k, len(cand)))  # keep at least 1 token
            keep_core = sorted(order[:k])
            kept = [ids[0]] + [ids[i] for i in keep_core] + [ids[-1]]
            kept_text = self.tokenizer.decode(kept, skip_special_tokens=True)
            p_keep = float(self.predict_fn(kept_text, target_label))
            out[r] = p_orig - p_keep  # lower is better
        return out
    
    def evaluate(self, explanations, labels=None):
        per_sample = []

        comp_collect = {r: [] for r in self.ratios}
        suff_collect = {r: [] for r in self.ratios}
        comp_auc_vals, comp_auc_norm_vals = [], []
        suff_auc_vals, suff_auc_norm_vals = [], []

        for ex in explanations:
            text   = ex.get("text", "")
            tokens = ex.get("tokens", [])
            imps   = self.get_importances(ex)
            yhat   = ex.get("pred_label")

            if yhat is None:
                # fall back: compute yhat by calling predict_fn and taking argmax over classes is not supported here.
                # For correctness, please store 'pred_label' in explanations.
                continue

            comp = self.comprehensiveness(text, tokens, imps, yhat)
            suff = self.sufficiency(text, tokens, imps, yhat)

            comp_auc, comp_auc_norm = self.curve_auc(comp)
            suff_auc, suff_auc_norm = self.curve_auc(suff)

            per_sample.append({
                "id": ex.get("sample_id", ex.get("id")),
                "label": ex.get("true_label"),
                "pred_label": yhat,
                "comprehensiveness": comp,
                "sufficiency": suff,
                "comp_auc": comp_auc,
                "comp_auc_norm": comp_auc_norm,
                "suff_auc": suff_auc,
                "suff_auc_norm": suff_auc_norm
            })

            for r, v in comp.items():
                comp_collect[r].append(v)
            for r, v in suff.items():
                suff_collect[r].append(v)

            if not np.isnan(comp_auc):      
                comp_auc_vals.append(comp_auc)
            if not np.isnan(comp_auc_norm): 
                comp_auc_norm_vals.append(comp_auc_norm)
            if not np.isnan(suff_auc):      
                suff_auc_vals.append(suff_auc)
            if not np.isnan(suff_auc_norm): 
                suff_auc_norm_vals.append(suff_auc_norm)


        def summarize(bucket):
            out = {}
            for r in self.ratios:
                vals = bucket.get(r, [])
                if vals:
                    out[r] = {
                        "mean": float(np.mean(vals)), 
                        "std": float(np.std(vals)), 
                        "n": int(len(vals))
                    }
                else:
                    out[r] = {
                        "mean": float("nan"), 
                        "std": float("nan"), 
                        "n": 0
                    }
            return out

        def summarize_list(vals):
            if vals:
                return {
                    "mean": float(np.mean(vals)), 
                    "std": float(np.std(vals)), 
                    "n": int(len(vals))
                }
            return {
                "mean": float("nan"), 
                "std": float("nan"), 
                "n": 0
            }

        return {
            "per_sample": per_sample,
            "aggregate": {
                "comprehensiveness": summarize(comp_collect),
                "sufficiency": summarize(suff_collect),
                "comp_auc": summarize_list(comp_auc_vals),
                "comp_auc_norm": summarize_list(comp_auc_norm_vals),
                "suff_auc": summarize_list(suff_auc_vals),
                "suff_auc_norm": summarize_list(suff_auc_norm_vals),
            }
        }
