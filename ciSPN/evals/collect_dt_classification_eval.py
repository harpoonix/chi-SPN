import glob
import numpy as np

#ext_str = ""
ext_str = "_noInterventions"

for dataset in ["CHAIN", "ASIA", "CANCER", "EARTHQUAKE"]:
    for score in ["GiniIndex", "CausalLossScore"]:
        #print("CFG", dataset, score)
        file_names = glob.glob(f"../../experiments/E3/eval_logs/E3_{dataset}_DT_{score}{ext_str}/*.txt")

        accuracies = []
        for file_name in file_names:
            with open(file_name) as f:
                lines = f.readlines()
                accuracy = float(lines[-1].split(" ")[-1])*100
                accuracies.append(accuracy)

        mean = np.mean(accuracies)
        std = np.std(accuracies)
        #print(f"accs: {accuracies}")
        #print(f"min: {np.min(accuracies)} | max: {np.max(accuracies)}")
        print(f"{dataset} {score} mean|std:", f"{mean:.2f}, {std:.2f}")


print("======= SCIKIT =======")

for dataset in ["CHAIN", "ASIA", "CANCER", "EARTHQUAKE"]:
    for score in ["GiniIndex"]:
        print("CFG", dataset, score)
        file_names = glob.glob(f"../../experiments/E3/eval_logs/E3_{dataset}_DTSciKit_{score}{ext_str}/*.txt")

        accuracies = []
        for file_name in file_names:
            with open(file_name) as f:
                lines = f.readlines()
                accuracy = float(lines[-1].split(" ")[-1])*100
                accuracies.append(accuracy)

        mean = np.mean(accuracies)
        std = np.std(accuracies)
        #print(f"accs: {accuracies}")
        #print(f"min: {np.min(accuracies)} | max: {np.max(accuracies)}")
        print(f"{dataset} {score} mean|std:", f"{mean:.2f}, {std:.2f}")
