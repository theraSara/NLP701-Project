# Beyond the Black Box: A Comparative Study of Interpretability Methods for Transformer-Based Sentiment Classification

A reproducible framework designed to evaluate and compare post-hoc interpretability methods for NLP classifiers. This project benchmarks four popular techniques—Attention, Integrated Gradients (IG), LIME, and SHAP—on lightweight, fine-tuned Transformer models (TinyBERT, DistilBERT, ALBERT) for sentiment analysis across two datasets (SST-2, IMDB).

The primary focus is on the following evaluations:

* **Faithfulness**: Do the highlighted tokens truly matter to the model's prediction?

* **Stability**: Are the explanations robust and consistent when faced with small input changes?

The runner produces structured CSV results and optional plots.

## 1. Key Features

* **Turn-key Runner**: Execute Attention, IG, LIME, and SHAP across multiple Hugging Face models and datasets with a single command.

* **Faithfulness Metrics**: Measures Comprehensiveness (higher is better) and Sufficiency (lower drop is better).

* **Stability Metrics**: Spearman correlation under controlled input shuffles (higher is better).

* **Method-Agnostic Post-Processing**: Special tokens are filtered consistently across all methods followed by L1 normalization.

* **Batchable & GPU-Friendly**: Automatically utilizes CUDA if available.

## 2. Evaluation Scope

### Interpretability Methods

| Name                          | Description                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------- |
| **Attention**                 | Averages self-attention weights across heads/layers to assign token importance. |
| **Integrated Gradients (IG)** | Path-integrated gradients from a baseline to the input (via Captum).            |
| **LIME**                      | Local linear surrogate over perturbed texts to estimate token importance.       |
| **SHAP**                      | Approximate Shapley values for token contributions to the prediction.           |

