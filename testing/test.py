import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
import PIL

#DOESN'T WORK
os.chdir("..")
file_im = "data/images/imgs_part_1/PAT_8_15_820.png"
im = plt.imread(file_im)
mask = plt.imshow(im,cmap="gray")
im[mask==0]=0
plt.imsave("test.png",im)