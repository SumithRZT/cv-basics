import numpy as np
from scipy import ndimage
import cv2


# Smoothing images
def gaussian_filter(size, fwhm = 3, center=None):

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


# Useful for edge detection
def apply_gaussian_highpass_filter(size, image):
    lowpass = ndimage.gaussian_filter(image, size)
    high_pass = image - lowpass

    return high_pass


# Helps removing s&p king of noises
def median_filter(image, size):
    return cv2.medianBlur(image, size)


# Edge preserving smoothing
def bilateral_filter(image, size):
    return cv2.bilateralFilter(image, size)


# Sharpens the image
def sharpen_image(image, impulse):
    kernel = np.array([[-1, -1, -1], [-1, impulse, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)