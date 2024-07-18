import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def mse_false(img1, img2):
    return np.mean(np.sqrt((img1 - img2) ** 2))

def mae(img1, img2):
    return np.mean(np.abs(img1-img2))

def psnr(img1, img2):
    this_mse = np.mean((img1 - img2) ** 2)
    if this_mse == 0:
        return 100
    return 20 * np.log10(255. / np.sqrt(this_mse))


def ssim(y_true, y_pred, multichannel):
    return ssim_skimage(y_true, y_pred, multichannel=multichannel, data_range=255)
