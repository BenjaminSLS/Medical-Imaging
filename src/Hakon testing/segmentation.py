import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage import morphology

#Go to main branch
os.chdir("..")
os.chdir("..")

#Code
def apply_filters():
    """
    Applies the masks from the segmentation directory to their corresponding images 
    and saves the original image, mask and filtered image 
    in images2 folder in Hakon testing directory
    """
    directory = "segmentation/masks"
    for filename in os.listdir(directory):
        f = os.path.join(directory,filename) 
        if os.path.isfile(f): #Make sure junk isn't loaded
            #Load images
            mask = np.load(f)
            filename_split = filename.split("_")[:-1]
            patient = "_".join(filename_split)+".png"
            im = plt.imread(f"data/images/images/{patient}")
            
            #Filter images
            filtered_im = im
            filtered_im[mask==0]=0

            im = plt.imread(f"data/images/images/{patient}") #Load again, idk why
            
            #Make subplots for all images
            fig,axes = plt.subplots(ncols=3,figsize=(16,10))
            axes[0].imshow(im)
            axes[0].set_title("Original image")
            axes[1].imshow(mask,cmap="gray")
            axes[1].set_title("Mask")
            axes[2].imshow(filtered_im)
            axes[2].set_title("Filtered image")

            #Save subplot
            plt.savefig("src/Hakon testing/images_manual_segmentation/"+"_".join(filename_split)+"_mask.png")
            plt.close() #Close to prevent overload

def save_data():
    """
    Function to save lesion features into csv file
    Automatically goes through all masks in the segmentation directory and saves it into the metadata.csv file.
    """
    directory = "segmentation/masks"
    
    #testing avg difference between area and perimeter
    diff_sum = dict()
    count = dict()
    for i in range(2,50,2):
        diff_sum[i]=0
        count[i]=0

    count_2 = 0
    for filename in os.listdir(directory):
        count_2+=1
        if count_2 < 6:
            f = os.path.join(directory,filename) 
            if os.path.isfile(f): #Make only files are loaded - Might be unnecessary
                mask = np.load(f)
                filename_split = filename.split("_")[:-1]
                patient = "_".join(filename_split)+".png"
                im = plt.imread(f"data/images/images/{patient}")

                #Area and perimeter features
                for i in range(2,50,2):
                    area, perimeter = features(mask,i)   
                    #print(patient,"\nArea:",round(area,2),"\nPerimeter:",round(perimeter,2),"\n") #To test outputs
                    diff_sum[i] += area-perimeter
                    count[i] += 1
    for i in range(2,50,2):
        print(f"Average difference between area and perimeter for brushsize {i}: \n",diff_sum[i]/count[i])

def features(mask,brushsize=2):
    """
    Function to measure the area and perimeter of a given mask.
    Returns the area and perimeter of the skin lesion.
    """
    
    #Total image size
    total = mask.shape[0] * mask.shape[1] 

    #Area    
    area = np.sum(mask) / total #Find area and standardize

    #Perimeter
        #Brush
    brush = morphology.disk(brushsize) #Needs to be changed, area and perimeter are almost the same

        #Erode image - i.e. eat away the borders
    mask_eroded = morphology.binary_erosion(mask,brush)

        #Find perimeter
    perimeter_mask = mask - mask_eroded

        #Find perimeter value and standardize
    perimeter = np.sum(perimeter_mask) / total

    return area,perimeter

def main():
    #Functions to run
    save_data()

if __name__ == "__main__":
    main()