import os
import csv
import json
import time
import datetime
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from utils import extract_eval_logs, best_epoch_and_dev, write_dev_curve

class ModelTrainer:
    def __init__(self, model_name, dataset_name, output_dir):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir

    def load_data(self):
        if self.dataset_name == 'sst2':
            dataset = load_dataset("glue", "sst2")
        elif self.dataset_name == 'imdb':
            dataset = load_dataset("imdb")

            train_val = dataset['train'].train_test_split(test_size=0.1, seed=42)
            dataset['train'] = train_val['train']
            dataset['validation'] = train_val['test']
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        print(f"Loaded {self.dataset_name}")
        print(f"Train size: {len(dataset['train'])}")
        print(f"Validation size: {len(dataset['validation'])}")
        print(f"Test size: {len(dataset['test'])}")
        print(f"Columns: {dataset['train'].column_names}")

        return dataset
    
    def tokenize_dataset(self, dataset, tokenizer):
        max_length = 128 if self.dataset_name == 'sst2' else 512
        text_key = 'sentence' if self.dataset_name == 'sst2' else 'text'

        def tokenize_func(examples):
            tokenize = tokenizer(
                examples[text_key], 
                padding='max_length', 
                truncation=True, 
                max_length=max_length
            )
            tokenize['labels'] = examples['label']
            return tokenize
        
        tokenized = dataset.map(
            tokenize_func, 
            batched=True
        )

        print(f"After tokenization:")
        print(f"Columns: {tokenized['train'].column_names}")
        print(f"Sample keys: {list(tokenized['train'][0].keys())}")

        return tokenized
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
        }
        return metrics
    
    def train(self, epochs=5, batch_size=16, learning_rate=2e-5):
        print(f"Training {self.model_name} on {self.dataset_name} dataset...")
        
        t0 = time.time()
        
        # load dataset
        dataset = self.load_data()


        # handle sst2 test set without labels
        if self.dataset_name == 'sst2':
            # since sst2 test set has no labels, split the validation set into val/test
            val_test_split = dataset['validation'].train_test_split(test_size=0.5, seed=42)
            dataset['validation'] = val_test_split['train']
            dataset['test'] = val_test_split['test']

            print(f"Validation size: {len(dataset['validation'])}")
            print(f"Test size: {len(dataset['test'])}")

        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )

        tokenized_dataset = self.tokenize_dataset(dataset, tokenizer)

        # define training args
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_dir=f'{self.output_dir}/logs',
            logging_strategy="epoch",
            logging_steps=1,
            save_total_limit=2,
            fp16=False, # False for Mac, True for GPU 
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        # extract per-epoch eval logs, write curve artifacts
        eval_logs = extract_eval_logs(trainer)
        write_dev_curve(self.output_dir, self.model_name, self.dataset_name, eval_logs)
        best_epoch, best_dev_acc = best_epoch_and_dev(eval_logs, metric='eval_accuracy')

        # save the final model and tokenizer
        trainer.save_model(f"{self.output_dir}/final")
        tokenizer.save_pretrained(f"{self.output_dir}/final")

        # evaluate on test set
        test_results = trainer.evaluate(tokenized_dataset['test'])
        print("Test Results:", test_results)

        # save test results
        with open(f"{self.output_dir}/test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

        elapsed = time.time() - t0
        finished_at = datetime.datetime.now().isoformat(timespec='seconds')
        
        summary = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "best_epoch": best_epoch,
            "dev_acc": float(best_dev_acc) if best_dev_acc is not None else None,
            "test_acc": float(test_results.get('eval_accuracy', float('nan'))),
            "fit_time_seconds": round(elapsed, 2),
            "finished_at": finished_at
        }
        print("Summary: ", summary)
        
        return summary
    
SUMMARY_CSV = './models/summary.csv'
TRAINING_CONFIGS = [
    {
        'model':'distilbert-base-uncased',
        'dataset':'sst2',
        'output_dir':'./models/distilbert_sst2',
        'batch_size':32
    },
    {
        'model':'distilbert-base-uncased',
        'dataset':'imdb',
        'output_dir':'./models/distilbert_imdb',
        'batch_size':16
    },
    {
        'model':'huawei-noah/TinyBERT_General_4L_312D',
        'dataset':'sst2',
        'output_dir':'./models/tinybert_sst2',
        'batch_size':32
    },
    {
        'model':'huawei-noah/TinyBERT_General_4L_312D',
        'dataset':'imdb',
        'output_dir':'./models/tinybert_imdb',
        'batch_size':16
    },
    {
        'model':'albert-base-v2',
        'dataset':'sst2',
        'output_dir':'./models/albert_sst2',
        'batch_size':32
    },
    {
        'model':'albert-base-v2',
        'dataset':'imdb',
        'output_dir':'./models/albert_imdb',
        'batch_size':16
    },
]

def run_all(configs):
    fieldnames=[
        "model",
        "dataset",
        "best_epoch",
        "dev_acc",
        "test_acc",
        "fit_time_seconds",
        "finished_at",
        "output_dir"
    ]
    
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for cfg in configs:
        trainer = ModelTrainer(
            model_name=cfg['model'],
            dataset_name=cfg['dataset'],
            output_dir=cfg['output_dir']
        )

        summary = trainer.train(
            epochs=cfg.get('epochs', 3),
            batch_size=cfg.get('batch_size', 16),
            learning_rate=cfg.get('learning_rate', 2e-5)
        )

        summary["output_dir"] = cfg['output_dir']
        with open(SUMMARY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(summary)
        
        print(f"Appended summary line to {SUMMARY_CSV}")

if __name__ == "__main__":
    run_all(TRAINING_CONFIGS)
