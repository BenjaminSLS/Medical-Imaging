import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

#Go to main branch
os.chdir("..")
os.chdir("..")



#Code
directory = "segmentation/masks"
for filename in os.listdir(directory):
    f = os.path.join(directory,filename) 
    if os.path.isfile(f): #Make sure junk isn't loaded
        mask = np.load(f)
        filename_split = filename.split("_")[:-1]
        patient = "_".join(filename_split)+".png"
        im = plt.imread(f"data/images/images/{patient}")

        filtered_im = im
        filtered_im[mask==0]=0

        im = plt.imread(f"data/images/images/{patient}")
        fig,axes = plt.subplots(ncols=3,figsize=(16,10))
        axes[0].imshow(im)
        axes[0].set_title("Original image")
        axes[1].imshow(mask,cmap="gray")
        axes[1].set_title("Mask")
        axes[2].imshow(filtered_im)
        axes[2].set_title("Filtered image")

        plt.savefig("src/Hakon testing/images_manual_segmentation/"+"_".join(filename_split)+"_mask.png")
        plt.close()