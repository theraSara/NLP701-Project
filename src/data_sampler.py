import random
import pickle
import numpy as np
from pathlib import Path
from datasets import load_dataset


class DataSampler:
    def __init__(self, output_dir='./data/sampled'):
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
        
        print(f"Total test samples in {dataset_name}: {len(test_data)}")
        print(f"Sampling: {n_sampled} examples")

        samples = {
            'texts': [],
            'labels': [],
            'indices': []
        }

        for label in [0, 1]:
            label_indices = [i for i, ex in enumerate(test_data) if ex['label'] == label]
            print(f"Label {label} has {len(label_indices)} samples")

            n_per_class = n_sampled // 2
            n_to_sample = min(n_per_class, len(label_indices))
            sampled_indices = random.sample(label_indices, n_to_sample)

            for idx in sampled_indices:
                samples['texts'].append(test_data[idx][text_key])
                samples['labels'].append(test_data[idx]['label'])
                samples['indices'].append(idx)

        print(f"Total sampled: {len(samples['texts'])} exampels")
        print("Negatice labels (0): {samples['labels'].count(0)}")
        print("Positive labels (1): {samples['labels'].count(1)}")

        output_file = self.output_dir / f"{dataset_name}_samppled_{n_sampled}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(samples, f)

        print(f"Saved sampled data to: {output_file}")
        return samples
    
    def load_samples(self, dataset_name, n_sampled=500):
        input_file = self.output_dir / f"{dataset_name}_samppled_{n_sampled}.pkl"
        
        if not input_file.exists():
            print(f"Sample file not found: {input_file}")
            return self.sample_test_set(dataset_name, n_sampled)

        with open(input_file, 'rb') as f:
            samples = pickle.load(f)

        print(f"Loaded sampled data from: {input_file}")
        return samples
    
if __name__ == "__main__":
    sampler = DataSampler()

    print("Sampling SST-2 dataset")
    sampler.sample_data('sst2', n_sampled=500) # uncomment the one you want to sample, and comment the other one

    #print("Sampling IMDB dataset")
    #sampler.sample_data('imdb', n_sampled=500)
