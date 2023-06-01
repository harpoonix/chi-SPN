import glob
import numpy as np

eval_folder = "eval_logs"
models = [("ciSPN", "NLLLoss", ""), ("mlp", "MSELoss", "_noInterventions"), ("mlp", "causalLoss", "")]
datasets = ["CHAIN", "CHC", "ASIA", "CANCER", "EARTHQUAKE"]
dataset_labels = {
    "CHAIN": "CHAIN",
    "CHC": "Causal Health Class.",
    "ASIA": "ASIA",
    "CANCER": "CANCER",
    "EARTHQUAKE": "EARTHQUAKE",
}


seeds = [606, 1011, 3004, 5555, 12096]

results_mean = {}
results_std = {}


def experiment_stats(paths):
    accuracies = []
    for file_name in paths:
        with open(file_name) as f:
            lines = f.readlines()
            accuracy = float(lines[-1].split(" ")[-1])*100
            accuracies.append(accuracy)

    mean = np.mean(accuracies)
    std = np.std(accuracies)
    #print(accuracies)
    return mean, std


# read neural/circuit results
for dataset in datasets:
    results_mean[dataset] = {}
    results_std[dataset] = {}
    for model, loss, ext_str in models:
        file_names = [f"../../experiments/E1/{eval_folder}/E1_{dataset}_{model}_{loss}{ext_str}/{seed}.txt" for seed in seeds]
        #print(model, loss, dataset)
        mean, std = experiment_stats(file_names)
        results_mean[dataset][model + loss] = mean
        results_std[dataset][model + loss] = std

for model, loss, ext_str in models:
    print(model, loss, ext_str, [results_mean[dataset][model + loss] for dataset in datasets])

# add combinations

# value to pick:
max_results = {
    "CHAIN": ("all", 0),
    "CHC": ("$\\geq$ 1.0", 4),
    "ASIA": ("0.001", 0),
    "CANCER": ("0.001", 0),
    "EARTHQUAKE": ("all", 0),
}

results_info = {}
factors = ["0.001", "0.01", "0.1", "1.0", "10.0"]
for dataset in datasets:
    choice_label, choice_idx = max_results[dataset]
    results = []
    for idx, factor in enumerate(factors):
        file_names = [f"../../experiments/E1/{eval_folder}/E1_{dataset}_mlp_MSELoss_causalLoss_{factor}/{seed}.txt" for seed in seeds]
        mean, std = experiment_stats(file_names)
        results.append(mean)
        if idx == choice_idx:
            results_mean[dataset]["combined"] = mean
            results_std[dataset]["combined"] = std
            results_info[dataset+"combined"] = f"\\textsubscript{{{choice_label}}}"
    print(f"combined {dataset}: {results}")



# add BN results
for dataset in datasets:
    with open(f"../../experiments/E1/{eval_folder}/E1_{dataset}_BN_causal/0.txt") as f:
        lines = f.readlines()
        accuracy = float(lines[-1].split(" ")[-1]) * 100
    results_mean[dataset]["BN"] = accuracy
    results_std[dataset]["BN"] = None

print("")
print("=== TABLE ===")
print("")

models_ordered = [*models, ("combined", None, ""), ("BN", None, "")]
print(" & ".join([f"{a}+{b}" for a,b,_ in models_ordered]))

results_best = {}
for dataset in datasets:
    results_best[dataset] = np.max(list(results_mean[dataset].values()))

    line = [f"{dataset_labels[dataset]}"]
    for model, loss, ext_str in models_ordered:
        entry = model + ("" if loss is None else loss)
        mean = results_mean[dataset][entry]
        std = results_std[dataset][entry]
        best = results_best[dataset]

        mean_str = f"{mean:.2f}"
        if abs(mean - best) <= 0.015:
            mean_str = f"\\textbf{{{mean_str}}}"

        info_str = results_info.get(dataset+entry, None)
        if info_str is not None:
            mean_str += info_str
        if std is not None:
            mean_str += f" $\pm$ {std:.2f}"
        line.append(f"{mean_str}")
        #if std is not None:
        #    line.append(f"{std:.2f}")

    line_str = " & ".join(line) + " \\\\"
    print(line_str)
