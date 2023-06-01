from pathlib import Path
from PIL import Image

from figures.plot_config import get_plot_config
import shutil


# /experiments/E3/dt_plots/E3_CANCER_DT_CausalLossScore/606.png
datasets = ["CHAIN", "ASIA", "CANCER", "EARTHQUAKE"]
models = [
    ("DT_CausalLossScore", ""),
    ("DT_GiniIndex", "_noInterventions"),
    ("DTSciKit_GiniIndex", "_noInterventions")
]
model_save_names = [
    "cl",
    "gini_corr",
    "gini_scikit_corr"
]

for dataset in datasets:
    for (model, ext), model_save_name in zip(models, model_save_names):
        path = Path(f"../../experiments/E3/dt_plots/E3_{dataset}_{model}{ext}/606.pdf")
        #with Image.open(path) as im:
        #    im = im.convert('RGB')
        #    im.save(f"./{dataset.lower()}_{model_save_name}_606.jpg", dpi=(get_plot_config('dpi'), get_plot_config('dpi')), optimize=True, quality=95)
        shutil.copyfile(path, f"./{dataset.lower()}_{model_save_name}_606.pdf")
