import re
import json
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
from bertviz import head_view, model_view
from transformers import AutoModelForSequenceClassification, AutoTokenizer
    

class AttentionVisualizer:
    def __init__(self, model, tokenizer, model_name="Model"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model.eval()

        # detect model type for compatibility
        self.model_type = self._detect_model_type()
        print(f"Initialized AttentionVisualizer for {self.model_type} model.")

    def _detect_model_type(self):
        model_type = getattr(self.model, "config", None)
        return getattr(model_type, "model_type", "unknown").upper()
        
    # generate attention-based explanation
    def explain(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        # get the attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        if outputs.attentions is None:
            raise RuntimeError("This model did not return attention weights.")

        # extract attention from all layers and heads
        attention = outputs.attentions # tuple of (batch, heads, seq_len, seq_len)

        # calculate multiple attention aggregation methods
        cls_attention = torch.stack([layer[0].mean(0)[0] for layer in attention]).mean(0)

        # average across all positions (not just CLS)
        avg_attention = torch.stack([layer[0].mean(dim=(0,1)) for layer in attention]).mean(0)

        # max attention across layers
        max_attention = torch.stack([layer[0].mean(0)[0] for layer in attention]).max(0)[0]

        # average attention weights across all heads and layers
        # focus on [CLS] token attention to other tokens
        #attention_scores = torch.stack([layer[0].mean(0)[0] for layer in attention])
        # average across layers
        #attention_scores = attention_scores.mean(0) # Shape: (seq_len, seq_len)

        # get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        pred_label = outputs.logits.argmax().item()
        pred_probs = torch.softmax(outputs.logits, dim=-1)[0]
        confidence = pred_probs.max().item()

        att = {
            'tokens': tokens,
            'cls_scores': cls_attention.cpu().numpy(),
            'avg_scores': avg_attention.cpu().numpy(),
            'max_scores': max_attention.cpu().numpy(),
            'prediction': pred_label,
            'confidence': confidence,
            'pred_probs': pred_probs.cpu().numpy(),
            'full_attention': attention,
            'input_ids': inputs['input_ids'],
            'text': text
        }
        return att
    
    def generate_bertviz_visualizations(self, text, output_dir, sample_id="sample"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        attention = outputs.attentions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # head view, shows attention patterns for each head 
        print(f"Generating head view for {self.model_name}...")
        head_view_html = head_view(attention, tokens, html_action='return')
        with open(output_dir / f"{sample_id}_{self.model_name}_head_view.html", 'w') as f: 
            f.write(head_view_html.data)

        # model view, shows attention patterns across layers 
        print(f"Generating model view for {self.model_name}...")
        model_view_html = model_view(attention, tokens, html_action='return')
        with open(output_dir / f"{sample_id}_{self.model_name}_model_view.html", 'w') as f: 
            f.write(model_view_html.data)

        print(f"BertViz visualizations saved to {output_dir}")
        return head_view_html, model_view_html
    
    def create_figures(self, explanation, output_path=None, top_k=15):
        fig = plt.figure(figsize=(16,10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        tokens = explanation['tokens']
        cls_scores = explanation['cls_scores']
        pred_label = explanation['prediction']
        confidence = explanation['confidence']
        pred_probs = explanation['pred_probs']

        # filter out special tokens for display
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
        filtered = [(i, t.replace('##', ''), s) for i, (t, s) in enumerate(zip(tokens, cls_scores)) 
                    if t not in special_tokens]
        if not filtered:
            return None
        

        # 1. text highlight visualization (top)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_text_highlight(ax1, tokens, cls_scores, pred_label, confidence)

        # 2. top-k bar chart (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        n = min(top_k, len(filtered))
        filtered.sort(key=lambda x: x[2], reverse=True)
        top_tokens = [x[1] for x in filtered[:n]]
        top_scores = [x[2] for x in filtered[:n]]
        self._plot_bar_chart(ax2, top_tokens, top_scores, pred_label, confidence)

        # 3. attention heatmap (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_heatmap(ax3, tokens, cls_scores)

        # 4. prediction confidence (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_prediction_confidence(ax4, pred_probs, pred_label)

        # 5. attention distribution (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_attention_distribution(ax5, cls_scores, tokens)

        # overall title
        label_name = 'Positive' if pred_label == 1 else 'Negative'
        fig.suptitle(f"{self.model_name} - Attention Analysis\nPrediction: {label_name} (Confidence: {confidence:.2%})",
                     fontsize=16, fontweight='bold', y=0.98)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight',facecolor='white')
            print(f"Saved figures to {output_path}")

        plt.close()
        return fig
    
    def _plot_text_highlight(self, ax, tokens, scores, pred_label, confidence):
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        cmap = plt.cm.RdYlGn

        x_pos = 0
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']

        for token, score in zip(tokens, norm_scores):
            if token in special_tokens:
                continue

            display_token = token.replace('##', '')
            color = cmap(score)

            bbox_props = dict(boxstyle="round, pad=0.4", fc=color, ec='gray',
                              linewidth=0.5, alpha=0.8)
            ax.text(x_pos, 0.5, display_token, bbox=bbox_props, fontsize=10,
                    verticalalignment='center', horizontalalignment='left')
            x_pos += len(display_token) * 0.18 + 0.4

        ax.set_xlim(0, x_pos)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Token-Level Attention Visualization', fontsize=12, pad=10)

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15, fraction=0.03, aspect=40)
        cbar.set_label('Attention Weight', fontsize=9)

    def _plot_bar_chart(self, ax, tokens, scores, pred_label, confidence):
        colors = plt.cm.RdYlGn(np.linspace(0.4, 0.95, len(tokens)))
        bars = ax.barh(range(len(tokens)), scores, color=colors, edgecolor='gray', linewidth=0.5)

        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel('Attention Weight', fontsize=10, fontweight='bold')
        ax.set_title(f"Top {len(tokens)} Important Tokens", fontsize=11, fontweight='bold')

        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.002, i, f'{score:.3f}', va='center', fontsize=8)

        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_heatmap(self, ax, tokens, scores):
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        scores_matrix = scores_norm.reshape(1, -1)

        sns.heatmap(scores_matrix, xticklabels=[t.replace('##', '') for t in tokens],
                    yticklabels=['Attention'], cmap='RdYlGn', cbar_kws={'label':'Weight'},
                    linewidth=0.5, linecolor='white', ax=ax, vmin=0, vmax=1)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_title('Attention Weight Heatmap', fontsize=11, fontweight='bold')

    def _plot_prediction_confidence(self, ax, pred_probs, pred_label):
        labels = ['Negative', 'Positive']
        colors = ['#ff6b6b' if i != pred_label else '#51cf66' for i in range(len(labels))]

        bars = ax.bar(labels, pred_probs, color=colors, edgecolor='gray', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('Probability', fontsize=10, fontweight='bold')
        ax.set_title('Classification Confidence', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
        ax.set_axisbelow(True)

        for bar, prob in zip(bars, pred_probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{prob:.2%}', ha='center',
                    va='bottom', fontsize=10)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_attention_distribution(self, ax, scores, tokens):
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
        filtered_scores = [s for s, t in zip(scores, tokens) if t not in special_tokens]

        ax.hist(filtered_scores, bins=30, color='steelblue', edgecolor='black',
                alpha=0.7, linewidth=0.5)
        ax.axvline(np.mean(filtered_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(filtered_scores):.3f}')
        ax.axvline(np.median(filtered_scores), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(filtered_scores):.3f}')
        
        ax.set_xlabel('Attention Weight', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title('Attention Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

def _safe_id(x):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(x))

def _explanation_to_jsonable(exp: Dict[str, Any]) -> Dict[str, Any]:
    json_file = {
        "tokens": list(map(str, exp["tokens"])),
        "cls_scores": np.asarray(exp["cls_scores"]).astype(float).tolist(),
        "avg_scores": np.asarray(exp["avg_scores"]).astype(float).tolist(),
        "max_scores": np.asarray(exp["max_scores"]).astype(float).tolist(),
        "prediction": int(exp["prediction"]),
        "confidence": float(exp["confidence"]),
        "pred_probs": np.asarray(exp["pred_probs"].astype(float)).tolist(),
        "input_ids": np.asarray(exp["input_ids"][0]).astype(int).tolist(),
        "full_attention_shape": [tuple(a.shape) for a in exp["full_attention"]],
        "text_len": len(exp["text"])
    }
    return json_file

def model_comparison_table(explanations_dict, output_path=None):
    data = []

    for model_name, exp in explanations_dict.items():
        tokens = exp['tokens']
        scores = exp['cls_scores']

        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
        filtered = [(t.replace('##', ''), s) for t, s in zip(tokens, scores)
                    if t not in special_tokens]
        
        if filtered:
            sorted_tokens = sorted(filtered, key=lambda x: x[1], reverse=True)
            top_5_tokens = [t[0] for t in sorted_tokens[:5]]
            top_5_scores = [t[1] for t in sorted_tokens[:5]]

            data.append({
                'Model': model_name,
                'Prediction': 'Positive' if exp['prediction'] == 1 else 'Negative',
                'Confidence': f"{exp['confidence']:.2%}",
                'Top Token': top_5_tokens[0] if top_5_tokens else 'N/A',
                'Top Score': f"{top_5_scores[0]:.4f}" if top_5_scores else 'N/A',
                'Avg Attention': f"{np.mean([s for _, s in filtered]):.4f}",
                'Std Attention': f"{np.std([s for _, s in filtered]):.4f}",
                'Top 5 Tokens': ', '.join(top_5_tokens[:5])
            })

    df = pd.DataFrame(data)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path.with_suffix('.csv'), index=False)

        fig, ax = plt.subplots(figsize=(16, len(data) * 0.8 + 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='left',
                         loc='center', colColours=['lightgray'] * len(df.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4a90e2')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, len(data) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Model Comparison: Attention Analysis',
                  fontsize=14, fontweight='bold', pad=20)
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close()

        print(f"Comparison table saved to {output_path}")

    return df

def load_samples_and_visualize(pkl_path, model_configs, output_base, dataset_name, num_samples=3):
    with open(pkl_path, 'rb') as f:
        samples = pickle.load(f)
        print(f"Recieved Samples from the path: {pkl_path} successfully")

    if isinstance(samples, dict):
        keys = {k.lower(): k for k in samples.keys()}
        k_text  = next((keys[k] for k in ('text','texts','sentence','sentences','review','content') if k in keys), None)
        k_label = next((keys[k] for k in ('label','labels','target','y')                      if k in keys), None)
        k_idx   = next((keys[k] for k in ('index','indices','id','ids')                      if k in keys), None)
        assert k_text is not None, "Couldn't find a text field in the pickle."

        n_total = len(samples[k_text])
        n = min(num_samples, n_total)

        texts = samples[k_text][:n]
        labels = samples[k_label][:n] if k_label else [None] * n
        indices = samples[k_idx][:n] if k_idx else list(range(n))

        iterable = list(zip(indices, texts, labels))
        print(f"Processing {n} samples from {dataset_name} dataset")

    else: 
        n = min(num_samples, len(samples))
        iterable = list(zip(range(n), samples[:n], [None]*n))
    
    output_dir = Path(output_base) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(samples)} samples from {dataset_name} dataset")

    for sample_idx, (sample_id, sample_text, sample_label) in enumerate(iterable, 1):
        print(f"Sample {sample_idx} | id={sample_id} | label={sample_label}")
        print(f"Text: {sample_text[:200]}")
        
        sid = _safe_id(sample_id)
        
        sample_output_dir = output_dir / f"sample_{sample_idx}_id_{sid}"
        sample_output_dir.mkdir(exist_ok=True)

        explanations = {}
        json_models: Dict[str, Any] = {}

        for model_name, model_path in model_configs.items():
            print(f"Processing {model_name}")

            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                attn_implementation="eager"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            visualizer = AttentionVisualizer(model, tokenizer, model_name)

            explanation = visualizer.explain(sample_text)
            explanations[model_name] = explanation

            bertviz_dir = sample_output_dir / 'bertviz'
            bv_sample_id = f"sample{sample_idx}_id{sid}"
            head_html_path = bertviz_dir / f"{bv_sample_id}_{model_name}_head_view.html"
            model_html_path = bertviz_dir / f"{bv_sample_id}_{model_name}_model_view.html"
            fig_path = sample_output_dir / f"{model_name}_analysis.png"
            
            visualizer.generate_bertviz_visualizations(
                sample_text,
                bertviz_dir,
                sample_id=bv_sample_id
            )

            visualizer.create_figures(
                explanation,
                output_path=fig_path
            )

            json_models[model_name] = {
                **_explanation_to_jsonable(explanation),
                "artifacts": {
                    "bertviz_head_view": str(head_html_path),
                    "bertviz_model_view": str(model_html_path),
                    "analysis_figure": str(fig_path)
                }
            }
        
        cmp_path = sample_output_dir / f"sample_{sample_id}_id{sid}_comparison"
        model_comparison_table(
            explanations,
            output_path=cmp_path
        )

        sample_json = {
            "dataset": dataset_name,
            "sample_index": int(sample_idx),
            "sample_id": sample_id,
            "sample_id_safe": sid,
            "label": None if sample_label is None else int(sample_label),
            "text": sample_text,
            "paths": {
                "folder": str(sample_output_dir),
                "comparison_csv": str(cmp_path.with_suffix(".csv")),
                "comparison_png": str(cmp_path.with_suffix(".png"))
            },
            "models": json_models
        }

        with open(sample_output_dir / f"sample_{sample_idx}_id{sid}", 'w', encoding="utf-8") as f:
            json.dump(sample_json, f, ensure_ascii=False, indent=2)

    print(f"All visualization completed and saved to {output_dir}")

def expand_models(templates, ds):
        out = {name: path.format(ds=ds) for name, path in templates.items()}
        return out

def main():
    DATASETS = {"imdb", "sst2"}
    
    SAMPLED_DATA = {
        "imdb": "data/sampled/imdb_sampled_500.pkl",
        "sst2": "data/sampled/sst2_sampled_436.pkl"
    }
    
    MODEL_TEMPLATES = {
        "DistilBERT": "models/distilbert_{ds}/final",
        "TinyBERT": "models/tinybert_{ds}/final",
        "ALBERT": "models/albert_{ds}/final"
    }

    OUTPUT = "./output/attention_analysis"
    NUM_SAMPLES = 15

    for ds in DATASETS:
        print(f"Running {ds}")
        load_samples_and_visualize(
            pkl_path=SAMPLED_DATA[ds],
            model_configs=expand_models(MODEL_TEMPLATES, ds),
            output_base=OUTPUT,
            dataset_name=ds,
            num_samples=NUM_SAMPLES
        )

    print("All datasets done")

if __name__ == "__main__":
    main()