import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
import random

def homomorphic_filter(channel):
    # Apply Homomorphic filter to a single channel
    imgLog = np.log1p(channel)
    M, N = imgLog.shape
    sigma = 20
    [X, Y] = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2
    gammaH = random.uniform(0.5, 2.0)  # generate random gammaH in range 1.0 to 2.0
    gammaL = random.uniform(0.0, 1.5)  # generate random gammaL in range 0.0 to 1.0
    H = np.exp(-gaussianNumerator / (2*sigma*sigma))
    H = gammaH + (gammaL - gammaH) * (1 - H)
    H = np.fft.ifftshift(H)
    If = np.fft.fft2(imgLog)
    Iout = np.real(np.fft.ifft2(If * H))
    Iout = np.exp(Iout) - 1
    # Normalize image intensity to 255
    Iout = cv2.normalize(Iout, None, 0, 255, cv2.NORM_MINMAX)
    Iout = np.uint8(Iout)
    return Iout

class HomomorphicFilter(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(HomomorphicFilter, self).__init__(always_apply, p)
        
    def apply(self, img, **params):
        for i in range(img.shape[2]):
            img[:, :, i] = homomorphic_filter(img[:, :, i])
        return img
