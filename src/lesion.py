import os
from matplotlib import pyplot as plt
import numpy as np
from numpy import int64
from skimage import morphology, color
from scipy.ndimage import rotate
from skimage.segmentation import slic,mark_boundaries
from skimage import feature
from skimage.measure import regionprops
import skimage.measure
import pandas as pd 
from scipy.ndimage.measurements import label
import time

imagesPath = "../data/images/images/"

class Lesion:
    lesion_id : str
    mask_source : any
    image_source : any
    mask: any
    top : int
    bottom : int
    left : int
    right : int
    image : any
    image_path : str
    mask_path : str
    filtered_img : any
    filtered_skin : any
    metadata: any
    
    
    

    def __init__(self, image_path,metadata) -> None:
        image_name_split = image_path.split("_")
        self.lesion_id = image_name_split[1]
        self.mask_path = "../segmentation/masks/" + image_path + "_mask.npy"
        self.image_path = "../data/images/images/" + image_path + ".png"
        
        try:
            self.mask = np.load(self.mask_path)
            self.mask[self.mask > 0] = 1
           
            self.image = plt.imread(self.image_path) # Returns a float from 0 to 1
            self.image = self.image[:,:,:3]
            self.metadata = metadata
  
        except Exception as e:
            print(e)
            print("Could not load image or mask")
            raise Exception("error")
            return None
        
    def isCancer(self) -> bool:
        cancers = ["BCC","MEL","SCC"]
     
        if self.metadata["diagnostic"].values[0] in cancers:
            return True
        return False

    def prepare_data(self):
        """
        Prepare data for training
        """
        
        
        features = self.metadata
        
        columns = ["patient_id","lesion_id","smoke","drink","background_father","background_mother","age","pesticide","gender","skin_cancer_history","cancer_history","has_piped_water","has_sewage_system","fitspatrick","region","diameter_1","diameter_2","diagnostic","itch","grew","hurt","changed","bleed","elevation","img_id","biopsed"]

        metadata = {}


        for col in columns:
        
            metadata[col] = self.metadata[col].values[0]


        metadata["compactness"] = self.get_compactness()
        metadata["rotation-asymmetry"] = self.get_rotation_asymmetry()
        metadata["asymmetry"] = self.get_asymmetry_feature()
        color = self.get_lesion_color_feature()
        metadata["hue-sd"] = color["hue"]

        # dataset_df = pd.DataFrame({"image_id":self.lesion_id,})
        # features = features[columns]
        
        

        df = pd.DataFrame.from_records([metadata])
        return (df,self.isCancer()) 
        
   
    def resize_center(self, buffer : int = 0): # This code will give a wrong image if there is more than 1 single lesion, so either we need to change it or make a new function for multiple lesions
        '''
        Resize the image by scanning for white pixels and cropping the image to the smallest possible size
        '''
        
        row_n,col_n=self.mask.shape
        
        left = -1
        right = -1
        top = -1
        bottom = -1
        
        # Top to bottom
        for td in range(row_n):
            if top == -1:
                if np.any(self.mask[td,:] == 1):
                    top = td
                    
            else:
                bottom = td
                if not np.any(self.mask[td,:] == 1):
                    break
                
      
        # Left to right
        for lr in range(col_n):

            if left == -1:
                if np.any(self.mask[:,lr] == 1):
                    left = lr
                    
            else:
                right = lr
                if not np.any(self.mask[:,lr] == 1):
                    break
                

        # Add buffer to the edges and make sure we don't go out of bounds
        self.top = top - buffer if top - buffer > 0 else 0
        self.bottom = bottom + buffer if bottom + buffer < row_n else row_n
        self.left = left - buffer if left - buffer > 0 else 0
        self.right = right + buffer if right + buffer < col_n else col_n

        #print(f'Top {self.top} Bottom {self.bottom} Left {self.left} Right {self.right}')
  
        self.mask_source = self.mask
        self.image_source = self.image
       
        # Resize the mask
        self.mask = self.mask[top:bottom, left:right]
        self.image = self.image[top:bottom,left:right]

        return self.mask, self.image

    def get_asymmetry_feature(self):
        """
        Find asymmetric index of mask by folding horizontally and vertically
        """
        mask = self.mask
        horizontal_flip = np.fliplr(mask)
    
        diff_horizontal_flip = mask - horizontal_flip 
        
        vertical_flip = np.flipud(mask)
        diff_vertical_flip = mask - vertical_flip

    
        
        diff_horizontal_area = np.count_nonzero(diff_horizontal_flip)
        diff_vertical_area = np.count_nonzero(diff_vertical_flip)

        if diff_horizontal_area == 0 or diff_vertical_area == 0:
            return 0
         
        assy_index = 0.5 * ((diff_horizontal_area / self._get_area())+(diff_vertical_area / self._get_area()))

        return assy_index
    
    def get_rotation_asymmetry(self):
        """
        Returns the rotation asymmetry of the lesion
        """
        mask = self.mask
        #Rotates the mask by 45 degrees
        rotated_mask = rotate(mask,angle=45)
        temp = self.mask

        self.mask = rotated_mask
        assymetry = self.get_asymmetry_feature()

        self.mask = temp
        
        return  assymetry

    def get_compactness(self):
        """
        Returns the compactness of the lesion
        """
        
        #Gets the area and perimeter of the lesion using functions _get_area and _get_perimeter
        area = self._get_area()
        perimeter = self._get_perimeter()


        #Calculates the compactness of the lesion
        compactness = perimeter ** 2 / (4 * np.pi * area)
                      
        return compactness
    
    def _get_area(self) -> int:
        '''
        Returns the area of the mask
        '''
        return np.sum(self.mask)
    
    def _get_perimeter(self, erosion_size : int = 1):
        """
        Returns the perimeter of the mask
        """
        mask = self.mask
        # Defines a disk brush of size erosion_size
        struct_el = morphology.disk(erosion_size)

        # Erodes the mask with the disk brush
        mask_eroded = morphology.binary_erosion(mask, struct_el)
        
        # Finds the size of the perimeter by subtracting the eroded mask from the original mask
        perimeter = np.sum(mask - mask_eroded)
       
    
        return perimeter
    
    def apply_mask_to_img(self):
        """
        Applies the mask to the image
        """

        #For resized images
        filtered_img = self.image.copy()
        filtered_img[self.mask==0] = [0,0,0]

        filtered_skin = self.image.copy()
        filtered_skin[self.mask==1] = [0,0,0]

        #For original images
        filtered_source_img = self.image_source.copy()
        
        filtered_source_img[self.mask_source==0] = [0,0,0]
        #print(filtered_source_img.shape)


        filtered_source_skin = self.image_source.copy()
        filtered_source_skin[self.mask_source==1] = [0,0,0]
       

        #print(filtered_source_skin.shape)


        #Save all images
        self.filtered_source_img = filtered_source_img
        self.filtered_source_skin = filtered_source_skin
        
        self.filtered_img = filtered_img
        self.filtered_skin = filtered_skin

    def get_skin_color_feature(self):
        """
        Finds the average skin color for each of the 3 color channels and returns the average as a tuple.
        Function doesn't take blue colors into account, we deem the effect to be negligible.
        """
        # Finds the total number of pixels in the image
        h,b = self.mask_source.shape
        total_pixel = h*b
        total_pixel_skin = total_pixel - np.sum(self.mask_source)
        
        # Finds the average skin color for each channel
        avg_r = np.sum(self.filtered_source_skin[:,:,0])/total_pixel_skin
        avg_g = np.sum(self.filtered_source_skin[:,:,1])/total_pixel_skin
        avg_b = np.sum(self.filtered_source_skin[:,:,2])/total_pixel_skin

        avg_col = np.asarray([[[avg_r,avg_g,avg_b,255]]])

        # fig,axs = plt.subplots(1,3,figsize=(9,9))
        # axs[0].imshow(self.filtered_source_skin)
        # axs[1].imshow(self.filtered_skin)
        # axs[2].imshow(avg_col)
        # plt.show()
        
        return (avg_r,avg_g,avg_b)

    def rgb_to_hsv(self,r,g,b):
        """"
        Changes rgb color to hsv
        """
        max_ = max(r,g,b)
        min_ = min(r,g,b)

        diff = max_ - min_

        if max_ == min_:
            h = 0
        elif max_ == r:
            h = (60 * ((g-b)/diff) + 360) % 360
        elif max_ == g:
            h = (60 * ((b-r)/diff) + 120) % 360
        elif max_ == b:
            h = (60 * ((r-g)/diff) + 240) % 360
        if max_ == 0:
            s = 0
        else:
            s = (diff/max_)*100
        v = max_*100
        return h, s, v
        

    def get_lesion_color_feature(self):
        """
        Returns the color features of the lesion using SLIC

        Uses imports skimage.segmentention.slic and skimage.segmentention.mark_boundaries
        to segment the image and mark the boundaries of the segments

        Then we find the average of each of the segments
        """
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3184884/ color extraction idea
        # https://biomedpharmajournal.org/vol12no1/melanoma-detection-in-dermoscopic-images-using-color-features/ another idea
 
        # Take out alpha channel (transparency)
        
        filtered_img = self.filtered_img.copy()[:,:,:3]

        # Segments the image using SLIC
        segments_slic = slic(filtered_img, n_segments=250, compactness=20,start_label=1,sigma=3)
        
        segments_slic_color = color.label2rgb(segments_slic, filtered_img, kind='avg')
        # fig2,axs2 = plt.subplots(1,2,figsize=(10,10))
        # axs2[0].imshow(filtered_img)
        # axs2[1].imshow(segments_slic_color)
        # plt.show()

        regions = regionprops(segments_slic,intensity_image=filtered_img)

        mean_intensity = [region.mean_intensity for region in regions]

        color_intensity = [mean for mean in mean_intensity if sum(mean) != 0]

        color_mean_hsv = [self.rgb_to_hsv(col[0],col[1],col[2]) for col in color_intensity]

        color_mean_hue = [hsv[0] for hsv in color_mean_hsv]
        color_mean_sat = [hsv[1] for hsv in color_mean_hsv]
        color_mean_val = [hsv[2] for hsv in color_mean_hsv]

        # check these values
        hue_sd = np.std(np.array(color_mean_hue))
        sat_sd = np.std(np.array(color_mean_sat))
        val_sd = np.std(np.array(color_mean_val))

        
        q1 = np.quantile(color_mean_val, 0.25, interpolation='midpoint')
        q3 = np.quantile(color_mean_val, 0.75, interpolation='midpoint')
        iqr = q3 - q1

        return {"hue":hue_sd,"sat":sat_sd,"val":val_sd,"quantile":iqr}
           
    def __str__(self) -> str:
        return f'{self.lesion_id}'


def main():
    pass
    #lesions = load_lesions()
    #print(len(lesions))
    #Circular lesion
 
    
    # lesion = Lesion("PAT_9_17_80")
    # #np.savetxt("before.txt",lesion.mask,fmt="%d")
    # lesion.resize_center(buffer=10)
    # #np.savetxt("mask.txt",lesion.mask,fmt="%d")
    # print("Circle lesion \n")
    # print("Asymmetry: ",lesion.get_asymmetry_feature())
    # print("Asymmetry 2: ", lesion.get_rotation_asymmetry())
    # print("Compactness: ",lesion.get_compactness())
    # print("Image: ", lesion.image)

    # #Oval lesion
    # print("\nOval lesion \n")

    # lesion = Lesion("PAT_1257_887_828")
    # lesion.resize_center(buffer=10)
    # print("Asymmetry: ",lesion.get_asymmetry_feature())
    # print("Asymmetry 2: ", lesion.get_rotation_asymmetry())
    # print("Compactness: ",lesion.get_compactness())


    # print("\nColor feature \n")
  
    # lesion = Lesion("PAT_20_30_44")
    # lesion.resize_center()
    # lesion.apply_mask_to_img()
    # print(lesion.get_skin_color_feature())
    # print(lesion.get_lesion_color_feature())
    

if __name__ == '__main__':
    main()
    