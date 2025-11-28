import json
import pandas as pd
from pathlib import Path

def consolidate_results():
    models = ['distilbert', 'tinybert', 'albert'] 
    datasets = ['sst2', 'imdb']
    
    results = []
    for model in models:
        for dataset in datasets:
            result_file = Path(f'./models/{model}_{dataset}/test_results.json')

            if result_file.exists():
                with open(result_file, 'r') as f: 
                    metrics = json.load(f)

                results.append({
                    'Model': model.upper(),
                    'Dataset': dataset.upper(),
                    'Accuracy': metrics.get('eval_accuracy', 0),
                    'F1': metrics.get('eval_f1', 0),
                    'Precision': metrics.get('eval_precision', 0),
                    'Recall': metrics.get('eval_recall', 0),
                    'Loss': metrics.get('eval_loss', 0)
                })
            else: 
                print(f"Warning: {result_file} does not exist.")

    df = pd.DataFrame(results)

    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / 'model_performance_summary.csv', index=False)
    print(f"Saved to: {output_dir / 'model_performance_summary.csv'}")

    return df

if __name__ == "__main__":
    consolidate_results()