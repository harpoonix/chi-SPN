import matplotlib.pyplot as plt
from PIL import Image

from figures.plot_config import get_plot_config

fig, ax = plt.subplots(nrows=1, ncols=4, dpi=get_plot_config('dpi'), figsize=(8, 1.5))

baseCImgs = "../../experiments/E2/gradcam/"

cimg_id = 4 # id=3 & 4 - B intervention

part_str = "_part2"
paths = [
    "./imageStructure.png",
    baseCImgs + f"{cimg_id}_cnn_MSELoss_noInterventions{part_str}.png",
    baseCImgs + f"{cimg_id}_ciCNNSPN_NLLLoss{part_str}.png",
    baseCImgs + f"{cimg_id}_cnn_causalLoss{part_str}.png"
]
titles = ["Underlying Structure", "NN with MSE", "ciSPN", "NN with Causal Loss"]

def load_and_show(ax, path, title=None):
    with Image.open(path) as im:
        ax.imshow(im)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')

ax = ax.reshape(-1)

for i, (path, title) in enumerate(zip(paths, titles)):
    load_and_show(ax[i], path, title=title)


#plt.tight_layout(w_pad=0.6, h_pad=0.05)
plt.tight_layout(rect=[-0.018,-0.1, 1.018, 1])
#plt.tight_layout(pad=0.8)
#plt.tight_layout()
#plt.subplots_adjust(wspace=0, hspace=0.5)
plt.savefig("./figure_gradCAM.jpg", dpi=get_plot_config('dpi'))
plt.savefig("./figure_gradCAM.pdf", dpi=get_plot_config('dpi'))
plt.savefig("./figure_gradCAM.eps", format="eps", dpi=get_plot_config('dpi'))
