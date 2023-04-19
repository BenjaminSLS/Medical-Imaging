import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data

from skimage.filters import gaussian
from skimage.segmentation import active_contour
import os

os.chdir("..")
img = plt.imread("data/images/imgs_part_1/PAT_8_15_820.png")
img = rgb2gray(img[:,:,:3])

print(img.shape)