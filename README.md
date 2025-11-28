# Beyond the Black Box: A Comparative Study of Interpretability Methods for Transformer-Based Sentiment Classification

A reproducible framework designed to evaluate and compare post-hoc interpretability methods for NLP classifiers. This project benchmarks four popular techniques—Attention, Integrated Gradients (IG), LIME, and SHAP—on lightweight, fine-tuned Transformer models (TinyBERT, DistilBERT, ALBERT) for sentiment analysis across two datasets (SST-2, IMDB).

The primary focus is on the following evaluations:

* **Faithfulness**: Do the highlighted tokens truly matter to the model's prediction?

* **Stability**: Are the explanations robust and consistent when faced with small input changes?

The runner produces structured CSV results and optional plots.

## Datasets and Models

| Resource     | URL                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| SST-2 (GLUE) | [https://huggingface.co/datasets/stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2)                                 |
| IMDB         | [https://huggingface.co/datasets/stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)                                 |
| DistilBERT   | [https://huggingface.co/docs/transformers/en/model_doc/distilbert](https://huggingface.co/docs/transformers/en/model_doc/distilbert) |
| TinyBERT     | [https://github.com/yinmingjun/TinyBERT](https://github.com/yinmingjun/TinyBERT)                                                     |
| ALBERT       | [https://huggingface.co/docs/transformers/en/model_doc/albert](https://huggingface.co/docs/transformers/en/model_doc/albert)         |


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

### Evaluation Metrics

| Name                  | Direction            | Definition                                                                                                             |
| --------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Comprehensiveness** | Higher is better (↑) | Drop in predicted probability when removing the top-k% important tokens.                                               |
| **Sufficiency Drop**  | Lower is better (↓)  | Drop in predicted probability when retaining only the top-k%; smaller drop implies the selected tokens are sufficient. |
| **Stability**         | Higher is better (↑) | Spearman correlation between original and perturbed explanation scores under controlled shuffles.                      |

## 3. Setup and Installation

### Prerequisities

* Python 3.8+

* PyTorch (GPU/CUDA recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data and Model Requirements

#### Datasets

This framework expects datasets in CSV format:

#### Expected Path

```bash
data/permutated100/sst2_shuffled_100.csv
```

#### Required Columns

`texts`, `labels`, `indices`

#### Model Checkpoints

All models must follow the Hugging Face format (with `config.json`, `pytorch_model.bin`, tokenizer files).
| Model Name         | Expected Path                    |
| ------------------ | -------------------------------- |
| TinyBERT (SST-2)   | `./models/tinybert_sst2/final`   |
| DistilBERT (SST-2) | `./models/distilbert_sst2/final` |
| ALBERT (SST-2)     | `./models/albert_sst2/final`     |
| TinyBERT (IMDB)    | `./models/tinybert_imdb/final`   |
| DistilBERT (IMDB)  | `./models/distilbert_imdb/final` |
| ALBERT (IMDB)      | `./models/albert_imdb/final`     |


## 4. Project Structure

```bash
.
├── data/
│   └── permutated100/
│       ├── imdb_shuffled_100.csv   
│       └── sst2_shuffled_100.csv
│   └── sampled/
│       ├── imdb_sampled_500.pkl   
│       └── sst2_sampled_500.pkl
├── src/
│   └── eval/
│       ├── eval_faithfulness.py             
│       └── eval_stability.py                 
│   └── interpretaibility_methods/
│       ├── attention.py             
│       ├── ig.py                    
│       ├── lime.py                  
│       ├── main.py                  # main evaluation runner (CLI)
│       ├── shap.py                  
│       └── utils.py                 
├── data_sampler.py
├── finetuning_results.py
├── metric.ipynb                     # Some Testing of the Evaluations 
├── shuffle_data.py
├── utils.py
├── models/
│   ├── tinybert_sst2/final/
│   ├── distilbert_sst2/final/
│   ├── albert_sst2/final/
│   ├── tinybert_imdb/final/
│   ├── distilbert_imdb/final/
│   └── albert_imdb/final/
└── results/                         # CSVs + figures 
```

Both data and results folders can be found [here]. (https://mbzuaiac-my.sharepoint.com/:f:/g/personal/sara_alhajeri_mbzuai_ac_ae/EqCEyEizpIBJiSQEkLME4nwB4FD1LqyjhkLyYH7OssnYuw?e=utslLP)
## 5. Running the Evaluation

The entry point is:

```bash
src/interpretaibility_methods/main.py
```

### Run All Methods

```bash
python src/interpretaibility_methods/main.py
```

### Run Selective Methods

Specify method keys (`attention`, `ig`, `lime`, `shap`):

```bash
python src/interpretaibility_methods/main.py lime shap
```

## 6. References

Key Literature
| Resource                                             | URL                                                                  |
| ---------------------------------------------------- | -------------------------------------------------------------------- |
| Integrated Gradients — Sundararajan et al., 2017     | [https://arxiv.org/abs/1703.01365](https://arxiv.org/abs/1703.01365) |
| LIME — Ribeiro et al., 2016                          | [https://arxiv.org/abs/1602.04938](https://arxiv.org/abs/1602.04938) |
| SHAP — Lundberg & Lee, 2017                          | [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874) |
| ERASER (faithfulness metrics) — DeYoung et al., 2020 | [https://arxiv.org/abs/1911.03429](https://arxiv.org/abs/1911.03429) |
| Stability / Sensitivity — Yin et al., 2022           | [https://arxiv.org/abs/2203.00847](https://arxiv.org/abs/2203.00847) |


