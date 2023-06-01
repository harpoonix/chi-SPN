import shutil

dataset = "CHAIN"
models = [
    #("DT_GiniIndex", ""),
    ("DT_GiniIndex", "_noInterventions"),
    #("DTSciKit_GiniIndex", "")
    ("DTSciKit_GiniIndex", "_noInterventions"),
    ("DT_CausalLossScore", "")
]

paths = [(f"../../experiments/E3/dt_plots/E3_{dataset}_{model}{ext}/606.svg", model, ext) for model, ext in models]


for path, model, ext in paths:
    shutil.copyfile(path, f"./{model}{ext}.svg")
