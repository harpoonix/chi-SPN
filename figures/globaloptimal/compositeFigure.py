import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from figures.plot_config import get_plot_config


plt.figure(dpi=get_plot_config('dpi'))

img = mpimg.imread('globalOptimal.png')[::-1,:,:]
plt.imshow(img)
w, h, c = img.shape

#fx = 0.30
#fy = 0.35
fx = 0.12
fy = 0.14


def add_gradient(im):
    y_min = 1330
    y_grad = 1420
    y_cut = 1780
    x_min = 830
    x_max = 2050

    print(im[y_grad, x_min, :])

    #im = np.asarray(im)
    for y in range(y_min, y_grad):
        a = 1 - ((y - y_min) / (y_grad - y_min))
        im[y, x_min:x_max, 3] = a * (im[y, x_min:x_max, 2] != 0)


    im[y_grad:y_cut, x_min:x_max, :] = 0
    #mpimg.
    #return PIL.Image.fromarray(np.uint8(im))
    return im

#imgA = mpimg.imread('SPNFull.png')
imgA = add_gradient(mpimg.imread('SPNFull.png'))
imgB = add_gradient(mpimg.imread('SPNMAP.png'))
iw, ih, ic = imgA.shape


iw *= fx
ih *= fy

# 200 / 92

offY = 0

offX = 100
plt.imshow(imgA, extent=[0 +offX, iw+offX, 0+offY, ih+offY])
#bottle_resized = resize(bottle, (140, 54))

offX = 1970
plt.imshow(imgB, extent=[0 +offX, iw+offX, 0+offY, ih+offY])

hpad= 50
wpad= 100
plt.ylim(hpad-50,w-hpad)
plt.xlim(wpad,h-wpad)
plt.tight_layout(h_pad=0.01, w_pad=0.01)
plt.axis('off')

plt.savefig("./figure_globalOptimal.jpg", bbox_inches='tight', dpi=get_plot_config('dpi'))
plt.savefig("./figure_globalOptimal.png", bbox_inches='tight', dpi=get_plot_config('dpi'))
plt.show()
