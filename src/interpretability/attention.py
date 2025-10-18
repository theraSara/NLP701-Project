import torch 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
    

class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

        # detect model type for compatibility
        self.model_type = self._detect_model_type()
        print(f"Initialized AttentionVisualizer for {self.model_type} model.")

    def _detect_model_type(self):
        model_name = self.model.__class__.__name__.lower()
        if 'distilbert' in model_name:
            return 'DistilBERT'
        elif 'albert' in model_name:
            return 'ALBERT'
        elif 'bert' in model_name:
            return 'BERT'
        else:
            return 'Unknown'
        
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

        # extract attention from all layers and heads
        attention = outputs.attentions # tuple of (batch, heads, seq_len, seq_len)

        # average attention weights across all heads and layers
        # focus on [CLS] token attention to other tokens
        attention_scores = torch.stack([layer[0].mean(0)[0] for layer in attention])
        # average across layers
        attention_scores = attention_scores.mean(0) # Shape: (seq_len, seq_len)

        # get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        pred_label = outputs.logits.argmax().item()
        confidence = torch.softmax(outputs.logits, dim=-1).max().item()

        att = {
            'tokens': tokens,
            'scores': attention_scores.cpu().numpy(),
            'prediction': pred_label,
            'confidence': confidence,
            'full_attention': attention
        }
        return att
    
    def print_top_tokens(self, explanation, top_k=10):
        tokens = explanation['tokens']
        scores = explanation['scores']
        pred_label = explanation['prediction']
        confidence = explanation['confidence']

        # filter out special tokens for display
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
        filtered_indices = [i for i, t in enumerate(tokens) if t not in special_tokens]
        if not filtered_indices:
            print("No non-special tokens")
            return []
        
        # get top-k from filtered
        filtered_scores = scores[filtered_indices]
        n = min(top_k, len(filtered_indices))
        top_k_filtered = np.argsort(filtered_scores)[-n:][::-1]
        # map back to original indices
        top_indices = [filtered_indices[i] for i in top_k_filtered]

        print(f"Prediction: {pred_label} (Confidence: {confidence:.3f})")
        print(f"Top {n} Important Tokens:")
        print(f"{'Rank':<6} {'Token':<20} {'Attention Score':<10}")
        
        out =[]
        for rank, idx in enumerate(top_indices, 1):
            token = tokens[idx].replace('##', '')
            score = scores[idx]
            print(f"{rank:<6} {token:<20} {score:.4f}")
            out.append((tokens[idx], scores[idx]))
        
        return out

    def visualize_text_highlight(self, explanation, output_path=None, max_tokens=50):
        tokens = explanation['tokens']
        scores = explanation['scores']
        pred_label = explanation['prediction']
        confidence = explanation['confidence']

        # limit to first max_tokens
        tokens = tokens[:max_tokens]
        scores = scores[:max_tokens]

        # normalize scores
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        # create figure
        fig, ax = plt.subplots(figsize=(16, 3))

        # color map: low importance (light) to high importance (dark)
        cmap = plt.cm.RdYlGn

        # plot each token with background color based on attention score
        x_pos = 0
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
        
        for i, (token, score) in enumerate(zip(tokens, norm_scores)):
            # skip special tokens 
            if token in special_tokens:
                continue

            # clean token 
            display_token = token.replace('##', '')

            # color based on importance
            color = cmap(score)

            # add text with colored background
            bbox_props = dict(
                boxstyle="round,pad=0.3",
                fc=color,
                ec='none',
                alpha=0.7
            )

            ax.text(
                x_pos,
                0.5,
                display_token,
                bbox=bbox_props,
                fontsize=11,
                verticalalignment='center',
                horizontalalignment='left'
            )

            # adjust spacing
            x_pos += len(display_token) * 0.15 + 0.3

        label_name = 'Positive' if pred_label == 1 else 'Negative'
        ax.set_title(f"Attention Visualization\nPrediction: {label_name} (Confidence: {confidence:.2%})", 
                    fontsize=14, fontweight='bold', pad=20)
        # remove axes
        ax.set_xlim(0, x_pos)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                            pad=0.1, fraction=0.05, aspect=30)
        cbar.set_label('Attention Score (Importance)', fontsize=10)
        
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention visualization to {output_path}")
        
        plt.close()
        return fig

    def visualize_heatmap(self, explanation, output_path=None, max_tokens=30):
        tokens = explanation['tokens']
        scores = explanation['scores']
        pred_label = explanation['prediction']
        confidence = explanation['confidence']

        # limit to first max_tokens 
        tokens = tokens[:max_tokens]
        scores = scores[:max_tokens]

        # normalize scores 
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        # create figure
        fig, ax = plt.subplots(figsize=(14, 2))

        # reshape for heatmap
        scores_matrix = scores_norm.reshape(1, -1) 

        # create heatmap
        sns.heatmap(
            scores_matrix,
            xticklabels=[t.replace('##', '') for t in tokens],
            yticklabels=['Attention'],
            cmap='RdYlGn',
            cbar_kws={'label': 'Importance Score'},
            linewidths=0.5,
            linecolor='white',
            ax=ax,
            vmin=0,
            vmax=1    
        )

        # rotate x-axis labels
        ax.set_xticklabels(
            ax.get_xticklabels(), 
            rotation=45, 
            ha='right'
        )

        # add title
        label_name = 'Positive' if pred_label == 1 else 'Negative'
        ax.set_title(f'Attention Importance Heatmap\nPrediction: {label_name} (Confidence: {confidence:.2%})',
                    fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {output_path}")

        plt.close()
        return fig
    
    def visualize_bar(self, explanation, output_path=None, top_k=15):
        tokens = explanation['tokens']
        scores = explanation['scores']
        pred_label = explanation['prediction']
        confidence = explanation['confidence']

        # filter out special tokens 
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
        filtered = [(t.replace('##', ''), s) for t, s in zip(tokens, scores) 
                    if t not in special_tokens]
        
        if not filtered:
            raise ValueError("No non-special tokens to display in bar chart")
        
        # cap top_k to available tokens
        n = min(top_k, len(filtered))

        # sort and get top-k
        filtered.sort(key=lambda x: x[1], reverse=True)
        top_tokens, top_scores = zip(*filtered[:n])

        # create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, n))
        bars = ax.barh(range(n), top_scores, color=colors)

        ax.set_yticks(range(n))
        ax.set_yticklabels(top_tokens)
        ax.set_xlabel('Attention Score (Importance)', fontsize=11)
        ax.set_ylabel('Tokens', fontsize=11)

        label_name = 'Positive' if pred_label == 1 else 'Negative'
        ax.set_title(f'Top {n} Important Tokens by Attention\nPrediction: {label_name} (Confidence: {confidence:.2%})',
                    fontsize=12, fontweight='bold', pad=15)
        
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(score + 0.001, i, f'{score:.3f}',
                    va='center', fontsize=9)
            
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # invert y-axis
        ax.invert_yaxis()

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention bar chart to {output_path}")

        plt.close()
        return fig
    
    def create_comparison_visualization(self, explanation_dict, output_path=None, max_tokens=20):
        n_examples = len(explanation_dict)
        fig, axes = plt.subplots(n_examples, 1, figsize=(14, 2 * n_examples))

        if n_examples == 1:
            axes = [axes]

        for ax, (name, explanation) in zip(axes, explanation_dict.items()):
            tokens = explanation['tokens'][:max_tokens]
            scores = explanation['scores'][:max_tokens]
            pred_label = explanation['prediction']
            confidence = explanation['confidence']

            # normalize scores
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

            # heatmap
            scores_matrix = scores_norm.reshape(1, -1)
            sns.heatmap(
                scores_matrix,
                xticklabels=[t.replace('##', '') for t in tokens],
                yticklabels=[name],
                cmap='RdYlGn',
                cbar_kws={'label': 'Importance Score'},
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                vmin=0,
                vmax=1    
            )

            ax.set_xticklabels(
                ax.get_xticklabels(), 
                rotation=45, 
                ha='right'
            )

            label_name = 'Positive' if pred_label == 1 else 'Negative'
            ax.set_title(f'{name} Model\nPrediction: {label_name} (Confidence: {confidence:.2%})',
                         fontsize=12, fontweight='bold')
            
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison visualization to {output_path}")

        plt.close()
        return fig
    
if __name__ == "__main__":
    print("Testing AttentionVisualizer...")
    model_path = "/Users/vanilla/Documents/courses/NLP701/project/NLP701-Project/models/distilbert_sst2/final"

    print(f"Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # create visualizer
    visualizer = AttentionVisualizer(model, tokenizer)

    # test text
    test_texts = [
        "The movie was absolutely fantastic with stunning visuals and a gripping storyline.",
        "Terrible waste of time. Completely boring and predictable.",
        "It was okay, nothing special but not terrible either."
    ]

    output_dir = Path("./outputs/attention_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations in {output_dir}...")

    for i, text in enumerate(test_texts, 1):
        print(f"Example {i}: {text[:50]}...")

        explanation = visualizer.explain(text)
        visualizer.print_top_tokens(explanation, top_k=5)
        visualizer.visualize_text_highlight(
            explanation, 
            output_path=output_dir / f"text_highlight_{i}.png"
        )
        visualizer.visualize_heatmap(
            explanation, 
            output_path=output_dir / f"heatmap_{i}.png"
        )
        visualizer.visualize_bar(
            explanation, 
            output_path=output_dir / f"bar_chart_{i}.png",
            #top_k=15
        )

    print("Creating comparison visualization...")
    explanations = {
        f"Example {i+1}": visualizer.explain(text)
        for i, text in enumerate(test_texts)
    }

    visualizer.create_comparison_visualization(
        explanations,
        output_path=output_dir / "comparison_visualization.png"
    )

    print("All tests completed.")