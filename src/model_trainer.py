import json
import evaluate
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback


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
        return dataset
    
    def tokenize_dataset(self, dataset, tokenizer):
        max_length = 128 if self.dataset_name == 'sst2' else 512

        def tokenize_func(examples):
            text_key = 'sentence' if self.dataset_name == 'sst2' else 'text' 
            tokenize = tokenizer(
                examples[text_key], 
                padding='max_length', 
                truncation=True, 
                max_length=max_length
            )
            return tokenize
        
        tokenized = dataset.map(tokenize_func, batched=True)
        return tokenized
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        matrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
        }
        return matrics
    
    def train(self, epochs=3, batch_size=32, learning_rate=2e-5):
        print(f"Training {self.model_name} on {self.dataset_name} dataset...")
        
        dataset = self.load_data()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )

        tokenized_dataset = self.tokenize_dataset(dataset, tokenizer)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=100,
            save_total_limit=2,
            fp16=False # False for Mac, True for GPU or otherwise
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

        trainer.save_model(f"{self.output_dir}/final")
        tokenizer.save_pretrained(f"{self.output_dir}/final")

        test_results = trainer.evaluate(tokenized_dataset['test'])
        print("Test Results:", test_results)

        with open(f"{self.output_dir}/test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)

        return test_results
    

# change the model_name and dataset_name as needed for each combination.txt
# change the batch_size from the combination.text too
if __name__ == "__main__":
    trainer = ModelTrainer(
        model_name="distilbert-base-uncased",
        dataset_name="sst2",
        output_dir="./models/distilbert_sst2"
    )
    results = trainer.train()


