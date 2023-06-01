import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from figures.plot_config import get_plot_config

plt.rcParams.update({
    'pdf.fonttype': 42, # force true type (default is Font 3)
    'ps.fonttype': 42,
    #"text.usetex": True,
    "font.family": "Vollkorn"
})

datasets = ["CHAIN", "CHC", "ASIA", "CANCER", "EARTHQUAKE"]
dataset_labels = ["Causal Chain", "CHC", "ASIA", "CANCER", "EARTHQUAKE"]
eval_folder = "eval_logs"
setup_name = "mlp_MSELoss_causalLoss"

#datasets = ["hiddenObject"]
#dataset_abrevs = ["hiddenObject"]
#loss = "MSELoss"
#eval_folder = "evals_img_interv_pos"
#setup_name = "simpleCNNModel"

seeds = [606, 1011, 3004, 5555, 12096]

plt.rcParams["mathtext.fontset"] = "cm"
fig, axs = plt.subplots(nrows=1, ncols=len(datasets), dpi=get_plot_config('dpi'), figsize=(10, 2.2))


def tsplot(ax, data, **kw):
    x = np.arange(data.shape[1])
    plt.xlim(x[0]-0.1, x[-1]+0.1)
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)

    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    ax.fill_between(x, mn, mx, alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.errorbar(x, est, sd, markersize=8, capsize=5) #, linestyle='None') #, marker='|')
    ax.margins(x=0)


for i, (dataset, dataset_label, ax) in enumerate(zip(datasets, dataset_labels, axs)):
    factors = ["0.001", "0.01", "0.1", "1.0", "10.0"]
    labels =  ["1e-3",  "1e-2", "0.1", "1.0", "10"]

    #labelsize = 20
    #matplotlib.rcParams.update({'font.size': 20})

    models = dict()

    print(f"Evaluating: {dataset}")

    model_accs_mean = []
    model_accs_std = []
    model_accs = []
    for factor in factors:
        file_names = [f"../../experiments/E1/{eval_folder}/E1_{dataset}_{setup_name}_{factor}/{seed}.txt" for seed in seeds]

        accuracies = []
        for file_name in file_names:
            with open(file_name) as f:
                lines = f.readlines()
                accuracy = float(lines[-1].split(" ")[-1])*100
                accuracies.append(accuracy)

        mean = np.mean(accuracies)
        std = np.std(accuracies)
        #print(f"accs: {}")
        print(f"alpha> {factor} :", f"{[f'{a:.2f}' for a in accuracies]} (mean|std): {mean:.3f}, {std:.3f}")
        #print(f"min: {np.min(accuracies)} | max: {np.max(accuracies)}")

        model_accs_mean.append(mean)
        model_accs_std.append(std)
        model_accs.append(accuracies)

    models["MLP"] = {"mean": model_accs_mean, "std": model_accs_std, "accs": np.array(model_accs).T}


    argmax = np.argmax(models["MLP"]["mean"])
    val_max = models["MLP"]["mean"][argmax]
    print(f"=== MAX {dataset} === at alpha: {factors[argmax]} [acc: {val_max}]")

    tsplot(ax, models["MLP"]["accs"], color="#40b3ef")

    #ax.legend([f'MLP {dataset_abrev}'], loc="upper right")



    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60)
    ax.set_xlabel("$\\alpha$")
    if i == 0:
        ax.set_ylabel("Accuracy (%)")
    ax.set_title(dataset_label)

plt.tight_layout()
plt.tight_layout(rect=[0.0,-0.05, 1.018, 1.05])

plt.savefig("./figure_combinedTraining.jpg", dpi=get_plot_config('dpi'))
plt.savefig("./figure_combinedTraining.pdf", dpi=get_plot_config('dpi'))
plt.savefig("./figure_combinedTraining.eps", format='eps')
plt.show()
