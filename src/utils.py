import os
import csv
import matplotlib.pyplot as plt

# return list of dicts with per-epoch eval metrics
def extract_eval_logs(trainer):
    history = trainer.state.log_history
    eval_logs = [row for row in history if 'eval_accuracy' in row and 'epoch' in row]
    return eval_logs

# pick best epoch by a mettric, return (best_epoch, best_value)
def best_epoch_and_dev(eval_logs, metric='eval_accuracy'):
    if not eval_logs:
        return None, None
    best_i = max(range(len(eval_logs)), key=lambda i: eval_logs[i].get(metric, float('-inf')))
    best = eval_logs[best_i]['epoch'], eval_logs[best_i].get(metric)
    return best

# write dev_curve.csv and dev_accuracy_vs_epoch.png
def write_dev_curve(output_dir, model_name, dataset_name, eval_logs):
    if not eval_logs:
        return
    csv_path = os.path.join(output_dir, "dev_curve.csv")
    filednames = ['epoch', 'eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall', 'eval_loss']
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, filednames=filednames)
        writer.writeheader()
        for row in eval_logs:
            writer.writerow({k: row.get(k) for k in filednames})

    # plot dev accuracy vs. epochs
    epochs = [r['epoch'] for r in eval_logs]
    accs = [r['eval_accuracy'] for r in eval_logs]
    png_path = os.path.join(output_dir, "dev_accuracy_vs_epochs.png")
    plt.figure()
    plt.plot(epochs, accs, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy")
    plt.title(f"{model_name} on {dataset_name}")
    plt.grid(True)
    best_i = max(range(len(accs)), key=lambda i: accs[i])
    plt.scatter([epochs[best_i]], [accs[best_i]], s=80)
    for e, a in zip(epochs, accs):
        plt.annotate(f"{a:.3f}", (e,a), textcoords="offset points", xytext=(0,6), ha='center',fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
