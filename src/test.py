import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray

os.chdir("..")
file_im = "data/images/imgs_part_1/PAT_8_15_820.png"
im = plt.imread(file_im)
mask = plt.imread(file_im)
plt.imshow(mask,cmap="gray")
mask = rgb2gray(mask)*256
im[mask==0]=0
#plt.imsave("test.png",im)