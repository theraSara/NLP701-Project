import random
import pickle
import numpy as np
from pathlib import Path
from datasets import load_dataset


class DataSampler:
    def __init__(self, output_dir='./sampled_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sample_data(self, dataset_name, n_sampled=500, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        if dataset_name == 'sst2':
            dataset = load_dataset("glue", "sst2")
            val_test_split = dataset['validation'].train_test_split(test_size=0.5, seed=seed)
            test_data = val_test_split['test']
            text_key = 'sentence'
        elif dataset_name == 'imdb':
            dataset = load_dataset("imdb")
            test_data = dataset['test']
            text_key = 'text'
        else: 
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        actual_n_samples = min(n_sampled, len(test_data))
        print(f"Requested {n_sampled}, but only {actual_n_samples} available")
        print(f"Sampling: {actual_n_samples} examples")

        by_label = {0: [], 1: []}
        for i, ex in enumerate(test_data):
            by_label[ex['label']].append(i)

        print(f"Label 0 has {len(by_label[0])} samples")
        print(f"Label 1 has {len(by_label[1])} samples")
        
        n_per_class = actual_n_samples // 2
        selected = []
        for label in [0, 1]:
            k = min(n_per_class, len(by_label[label]))
            selected += random.sample(by_label[label], k)

        shortfall = actual_n_samples - len(selected)
        if shortfall > 0:
            remaining = list(set(range(len(test_data))) - set(selected))
            extra = random.sample(remaining, min(shortfall, len(remaining)))
            selected += extra

        samples = {
            'texts': [],
            'labels': [],
            'indices': []
        }
        for idx in selected:
            samples['texts'].append(test_data[idx][text_key])
            samples['labels'].append(int(test_data[idx]['label']))
            samples['indices'].append(idx)

        assert len(samples['texts']) == len(samples['labels']) == len(samples['indices']) == len(selected)

        n_negative = samples['labels'].count(0)
        n_positive = samples['labels'].count(1)
        print(f"Total sampled: {len(samples['texts'])} exampels")
        print(f"Negative labels (0): {n_negative}")
        print(f"Positive labels (1): {n_positive}")

        output_file = self.output_dir / f"{dataset_name}_sampled_{actual_n_samples}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(samples, f)

        print(f"Saved sampled data to: {output_file}")
        return samples
    
    def load_samples(self, dataset_name, n_sampled=500):
        input_file = self.output_dir / f"{dataset_name}_sampled_{n_sampled}.pkl"
        
        if not input_file.exists() and dataset_name == 'sst2':
            alt = self.output_dir / f"{dataset_name}_sampled_436.pkl"
            if alt.exists():
                input_file = alt
                print(f"Using available sample file: {alt}")
            
        if not input_file.exists():
            print(f"Sample file not found: {input_file}")
            return self.sample_data(dataset_name, n_sampled)

        with open(input_file, 'rb') as f:
            samples = pickle.load(f)

        print(f"Loaded sampled data from: {input_file}")
        return samples
    
if __name__ == "__main__":
    sampler = DataSampler()

    print("Sampling SST-2 dataset")
    sst_samples = sampler.sample_data('sst2', n_sampled=100) 

    print("Sampling IMDB dataset")
    imdb_samples = sampler.sample_data('imdb', n_sampled=100)

    print("Summary:")
    print(f"SST-2: {len(sst_samples['texts'])} examples")
    print(f"IMDB: {len(imdb_samples['texts'])} examples")
