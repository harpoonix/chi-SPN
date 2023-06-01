import glob
import numpy as np

datasets = ["CHAIN", "ASIA", "CANCER", "EARTHQUAKE"]
#setups = [("DT", "CausalLossScore", ""), ("DT", "GiniIndex", "_noInterventions"), ("DTSciKit", "GiniIndex", "_noInterventions"), ("DT", "GiniIndex", ""), ("DTSciKit", "GiniIndex", "")]
#setup_labels = ("CDS", "Gini Index ours (corr)", "Gini Index SciKit(corr)", "Gini Index ours", "Gini Index SciKit")
setups = [("DT", "GiniIndex", "_noInterventions"), ("DTSciKit", "GiniIndex", "_noInterventions"), ("DT", "CausalLossScore", "")]
setup_labels = ("Gini Index ours (corr)", "Gini Index SciKit (corr)", "CDS")
results = {}

for dataset in datasets:
    results[dataset] = {}
    for base, score, ext in setups:
        file_names = glob.glob(f"../../experiments/E3/eval_logs/E3_{dataset}_{base}_{score}{ext}/*.txt")

        accuracies = []
        for file_name in file_names:
            with open(file_name) as f:
                lines = f.readlines()
                accuracy = float(lines[-1].split(" ")[-1])*100
                accuracies.append(accuracy)

        mean = np.mean(accuracies)
        std = np.std(accuracies)
        results[dataset][base+score+ext] = (mean, std)

print("")
print("=== TABLE ===")
print("")

print(" & ".join([f"{base}+{score}{ext}" for base, score, ext in setups]))

for dataset in datasets:

    line = [f"{dataset}"]
    for base, score, ext in setups:
        mean, std = results[dataset][base+score+ext]

        mean_str = f"{mean:.2f}"
        std_str = f"{std:.2f}"

        #line.append(mean_str)
        #line.append(std_str)
        line.append(f"{mean_str} $\pm$ {std_str}")

    line_str = " & ".join(line) + " \\\\"
    print(line_str)
