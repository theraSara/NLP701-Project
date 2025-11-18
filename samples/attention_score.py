import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@torch.no_grad()
def get_attention_and_prob(model_dir: str, text: str, device: str | None = None, max_length: int = 512):
    """
    Compute attention scores and prediction probability for a single text sequence.

    Args:
        model_dir: path to fine-tuned HF model
        text: input sentence or document
        device: 'cuda' or 'cpu' (auto-select if None)
        max_length: tokenizer truncation length

    Returns:
        attn_scores: list[float]  - averaged [CLS]→token attention (padding trimmed)
        pred_prob: float          - probability of the predicted class
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, output_attentions=True)
    model.to(device).eval()

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward with attention outputs
    outputs = model(**inputs, output_attentions=True)

    # --- Predicted class probability ---
    probs = softmax(outputs.logits, dim=-1)           # [1, num_labels]
    pred_prob = probs.max(dim=-1).values.item()       # float

    # --- Averaged [CLS]→token attention ---
    attn_stack = torch.stack(outputs.attentions, dim=0)   # [L, 1, H, S, S]
    attn_mean = attn_stack.mean(dim=(0, 2))               # mean over layers & heads → [1, S, S]
    cls_row = attn_mean[0, 0, :]                          # [S] → CLS attends to all tokens

    # Trim padding (attention_mask=1 for real tokens)
    seq_len = int(inputs["attention_mask"][0].sum().item())
    attn_scores = cls_row[:seq_len].detach().cpu().tolist()

    return attn_scores, pred_prob


# if __name__ == "__main__":
    # # Datasets and models
    # datasets = [
    #     {"name": "imdb", "path": 'sampled_data/imdb_sampled_100.pkl', "limit": 50},
    #     {"name": "sst2", "path": 'sampled_data/sst2_sampled_100.pkl', "limit": 50},
    # ]

    # models = [
    #     {"name": "tinybert_sst2",   "dir": "D:/master/NLP/models/tinybert_sst2/final"},
    #     {"name": "albert_imdb",     "dir": "D:/master/NLP/models/albert_imdb/final"},
    #     {"name": "distilbert_imdb", "dir": "D:/master/NLP/models/distilbert_imdb/final"},
    # ]

    # output_root = "extracted_data"

    # for model in models:
    #     for dataset in datasets:
    #         model_name = model["name"]
    #         model_path = model["dir"]
    #         ds_name = dataset["name"]
    #         ds_path = dataset["path"]
    #         limit = dataset.get("limit", None)

    #         print(f"\n=== Running {model_name} on {ds_name} ===")
    #         with open(ds_path, 'rb') as f:
    #             imdb_samples = pickle.load(f)
    #             print(f"Loaded samples from: {ds_path}")
    #         if limit:
    #             df = df.iloc[:limit]

    #         attn_scores_list, pred_prob_list = [], []
    #         for text in df["texts"].astype(str):
    #             attn_scores, pred_prob = get_attention_and_prob(model_path, text)
    #             attn_scores_list.append(json.dumps(attn_scores))
    #             pred_prob_list.append(pred_prob)

    #         df["texts__attn_scores"] = attn_scores_list
    #         df["texts__pred_prob"] = pred_prob_list

    #         out_path = f"{output_root}/{model_name}_{ds_name}_attn_prob.csv"
    #         df.to_csv(out_path, index=False, encoding="utf-8")
    #         print(f"Saved: {out_path}")