import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from PIL import Image
import io

from figures.plot_config import get_plot_config

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

#active_color = colors[0]
#active_color = "#55aaff"
active_color = "tab:blue"
active_light_color = "lightgrey"
inactive_color = "lightgrey"
trace_color = "darkorange"

gs = [(0.5, 6, 0.25), (7, 5, 0.55), (13, 2.5, 0.2)]

plt.figure(figsize=(12, 1.5), dpi=get_plot_config('dpi'))
plt.xlim(-8, 49)
plt.ylim(-0.01, 0.105)

def follow(gs, x_range, y_sum, start):
    m,s,w = gs

    xs = np.argmax(x_range>m)
    xe = np.argmax(x_range>start)

    y = y_sum[xs:xe]
    x = x_range[xs:xe]
    c=trace_color
    plt.plot(x, y, c=c, linestyle=None, linewidth=2)
    #plt.plot(x, y, c=c, linestyle="dashed", linewidth=2)
    plt.scatter(x[0], y[0], marker='x', s=50, color=c)
    plt.scatter(x[-1], y[-1], marker='|', s=60, color=c) #, zorder=2)

def calc_gaussian(x, mean, variance, factor=1.0):
    sigma = np.sqrt(variance)
    return stats.norm.pdf(x, mean, sigma)*factor


x_lin = np.linspace(-7.5, 18.5, 300)
def multi_plot(gs, y_off=0.0, **kwargs):
    my = 0
    y = np.zeros_like(x_lin)
    for m, v, w in gs:
        yy = calc_gaussian(x_lin, m, v)*w
        my = max(my, max(yy))
        y += yy
    plt.plot(x_lin, y + y_off, **kwargs)
    return my,y

my, y_sum = multi_plot(gs, c=active_color)

#h_off = 0.15
h_off = -0.01
for m, v, w in gs:
    vv = min(1.6 * v, 7)
    x = np.linspace(m-vv, m+vv, 100)
    plt.plot(x, calc_gaussian(x, m, v)*w + h_off, color=active_light_color)

off = 0.12

follow(gs[2], x_lin, y_sum, start=18)

plt.text(-7, off-0.03, "a) Convergence to local optimum")


x_off = 30
gs = [(x+x_off,y,w) for x,y,w in gs]
x_lin += x_off
"""
for i, (m, v, w) in enumerate(gs):
    vv = min(1.6 * v, 7)
    x = np.linspace(m-vv, m+vv, 100)
    if i == 1:
        kwargs = {"c": active_color}
    else:
        kwargs = {"c": inactive_color, "linestyle": 'dashed'}
    plt.plot(x, calc_gaussian(x, m, v)*w+h_off, **kwargs)
"""

my, y_sum = multi_plot([gs[1]], c=active_color) #c="blue")
for i, (m, v, w) in enumerate(gs):
    vv = min(1.6 * v, 7)
    x = np.linspace(m-vv, m+vv, 100)
    plt.plot(x, calc_gaussian(x, m, v)*w + h_off,
             color=active_light_color if i==1 else inactive_color,
             linestyle='dashed' if i!=1 else None)
#multi_plot([gs[1]], c=inactive_color, y_off=h_off)
#multi_plot([gs[0], gs[2]], c=inactive_color, linestyle='dashed', y_off=h_off)


follow(gs[1], x_lin, y_sum, start=x_off+18)

plt.text(-7+x_off, off-0.03, "b) Global optimal convergence")


def fig_to_numpy(fig, dpi):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=dpi)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr.copy()


plt.axis('off')
plt.tight_layout(h_pad=0, w_pad=0)

#plt.savefig("./globalOptimal.png", dpi=get_plot_config('dpi'))
#plt.show()


# add fading
im = fig_to_numpy(plt.gcf(), dpi=get_plot_config('dpi'))

def lerp(a,b,w):
    return a*(1-w) + b*w

y0 = 330
y1 = 330+150
x0 = 300
x1 = 2200
im[y0:y1, 20:x0, :3] = 255
im[y0:y1, 1800:x1, :3] = 255

x_min = x0
x_max = x0+60
for x in range(x_min, x_max):
    im[y0:y1, x, :3] = lerp(im[y0:y1, x, :3], 255, 1-((x - x_min) / (x_max-x_min)))
x_min = x1
x_max = x1+60
for x in range(x_min, x_max):
    im[y0:y1, x, :3] = lerp(im[y0:y1, x, :3], 255, 1-((x - x_min) / (x_max-x_min)))

im = Image.fromarray(im)
im.save("./globalOptimal.png", dpi=(get_plot_config('dpi'), get_plot_config('dpi')))

