import cv2
import numpy as np
from os.path import expanduser
import random

img = cv2.imread('test.jpg')


def noisy(noise_typ, image):

      if noise_typ == "gauss":
         row,col,ch= image.shape
         mean = 0
         var = 0.1
         sigma = 32
         gauss = np.random.normal(mean,sigma,(row,col,ch))
         gauss = gauss.reshape(row,col,ch)
         noisy = image + gauss
         return noisy

      elif noise_typ == "s&p":
         s_vs_p = 0.5
         amount = 0.008
         out = np.copy(image)
         # Salt mode
         num_salt = np.ceil(amount * image.size * s_vs_p)
         coords = [np.random.randint(0, i - 1, int(num_salt))
                 for i in image.shape]
         out[coords] = 1

         # Pepper mode
         num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
         coords = [np.random.randint(0, i - 1, int(num_pepper))
                 for i in image.shape]
         out[coords] = 0
         return out

      elif noise_typ == "poisson":
         vals = len(np.unique(image))
         vals = 0.8 ** np.ceil(np.log2(vals))
         noisy = np.random.poisson(image * vals) / float(vals)
         return noisy

      elif noise_typ =="speckle":
         row,col,ch = image.shape
         gauss = np.random.randn(row,col,ch)
         gauss = gauss.reshape(row,col,ch)
         noisy = image + image * gauss
         return noisy


def salt_pepper_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

noised = noisy("speckle", img)
home = expanduser("~")
cv2.imwrite(home+'/noise_added.jpg', noised)