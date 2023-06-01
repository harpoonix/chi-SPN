import matplotlib.pyplot as plt
from PIL import Image

from figures.plot_config import get_plot_config

fig, ax = plt.subplots(nrows=2, ncols=4, dpi=get_plot_config('dpi'), figsize=(4, 1.5))

paths = [
    "0.jpg",
    "1.jpg",
    "4.jpg",
    "147.jpg",
    "166.jpg",
    "197.jpg",
    "215.jpg",
    "249.jpg",
]

def load_and_show(ax, path, title=None):
    with Image.open(path) as im:
        ax.imshow(im)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')

ax = ax.reshape(-1)

for i, path in enumerate(paths):
    load_and_show(ax[i], path)


#plt.tight_layout(w_pad=1.1, h_pad=0.05)
plt.tight_layout(pad=0.8)
plt.savefig("./figure_hiddenObject.jpg", dpi=get_plot_config('dpi'))
plt.savefig("./figure_hiddenObject.pdf", dpi=get_plot_config('dpi'))
plt.savefig("./figure_hiddenObject.eps", format="eps", dpi=get_plot_config('dpi'))
