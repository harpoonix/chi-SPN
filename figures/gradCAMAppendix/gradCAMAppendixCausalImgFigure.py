import matplotlib.pyplot as plt
from PIL import Image

from figures.plot_config import get_plot_config

baseCImgs = "../../experiments/E2/gradcam/"

def make_triple(id):
    ext = "_part2"
    return [
        baseCImgs + f"{id}_cnn_MSELoss_noInterventions{ext}.png",
        baseCImgs + f"{id}_ciCNNSPN_NLLLoss{ext}.png",
        baseCImgs + f"{id}_cnn_causalLoss{ext}.png"
    ]

paths = [
    *make_triple(0),
    *make_triple(3),
    *make_triple(6),
    *make_triple(9),
]
titles = ["NN", "ciSPN", "NN Causal Loss", None, None, None, None, None, None, None, None, None]


fig, ax = plt.subplots(nrows=4, ncols=3, dpi=get_plot_config('dpi'), figsize=(5, 4.5))

def load_and_show(ax, path, title=None):
    with Image.open(path) as im:
        ax.imshow(im)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')

ax = ax.reshape(-1)

for i, (path, title) in enumerate(zip(paths, titles)):
    load_and_show(ax[i], path, title=title)


#plt.tight_layout(w_pad=1.1, h_pad=0.05)
plt.tight_layout(pad=0.7)
plt.savefig("./figure_appendix_gradCAMCausalImg.jpg", dpi=get_plot_config('dpi'))
plt.savefig("./figure_appendix_gradCAMCausalImg.pdf", dpi=get_plot_config('dpi'))
plt.savefig("./figure_appendix_gradCAMCausalImg.eps", format="eps", dpi=get_plot_config('dpi'))
