import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import filters
from skimage import morphology
import os

#Change directory
os.chdir("..")
os.chdir("..")

n = int(input()) #Amount of images

brush = struct_el = morphology.disk(6)

for _ in range(n):
    
    #Read images
    file_img = input()
    file_name = file_img.strip(".png").split("\\")
    img = plt.imread(file_img)
    #Remove 4th channel of PNG image
    im = img[:,:,:3]

    #Make gray
    im = rgb2gray(im)*256

    #Threshold for filtering
    threshold = filters.threshold_otsu(im)

    #Mask
    mask = im < threshold

    #Brush and cool stuff
    cool_mask = morphology.binary_opening(mask,brush)

    #Filter mask off
    img_filtered = img[:,:,:3]
    img_filtered[cool_mask==0]=0

    img = plt.imread(file_img)

    #Plot as subplot and save
    fig,axes=plt.subplots(ncols=3,figsize=(15,9))
    axes[0].imshow(img)
    axes[1].imshow(img_filtered)
    axes[2].imshow(cool_mask,cmap="gray")
    axes[0].set_title("Original image")
    axes[1].set_title("Filtered image")
    axes[2].set_title("Mask")

    plt.savefig("src/Hakon testing/images/" +file_name[-1] +"_Original_Mask.png")