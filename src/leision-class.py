from matplotlib import pyplot as plt
import numpy as np
from numpy import int64
from skimage import morphology
from scipy.ndimage import rotate
from skimage.segmentation import slic,mark_boundaries
from skimage import feature

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
    
    

    def __init__(self, image_path) -> None:
        image_name_split = image_path.split("_")
        self.lesion_id = image_name_split[1]
        self.mask_path = "../segmentation/masks/" + image_path + "_mask.npy"
        self.image_path = "../data/images/images/" + image_path + ".png"
        
        try:
            self.mask = np.load(self.mask_path)
            self.mask[self.mask > 0] = 1
            self.image = plt.imread(self.image_path)
        except:
            print("Could not load image or mask")
            return None
            
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

        print(f'Top {self.top} Bottom {self.bottom} Left {self.left} Right {self.right}')
  
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
        
        np.savetxt("mask_eroded.txt",mask_eroded,fmt="%d")

        # Finds the size of the perimeter by subtracting the eroded mask from the original mask
        perimeter = np.sum(mask - mask_eroded)
        print(f'Perimeter {perimeter}')
    
        return perimeter
    

    def get_average_skin_color(self):
        pass
        
    
    
    def apply_mask_to_img(self):
        """
        Applies the mask to the image
        """
        #For resized images
        filtered_img = self.image.copy()
        filtered_img[self.mask==0] = [0,0,0,255]
        filtered_skin = self.image.copy()
        filtered_skin[self.mask==1] = [0,0,0,255]

        #For original images
        filtered_source_img = self.image_source.copy()
        filtered_source_img[self.mask_source==0] = [0,0,0,255]
        filtered_source_skin = self.image_source.copy()
        filtered_source_skin[self.mask_source==1] = [0,0,0,255]

        self.filtered_source_img = filtered_source_img
        self.filtered_source_skin = filtered_source_skin
        self.filtered_img = filtered_img
        self.filtered_skin = filtered_skin


    def average_skin_color(self):
        """
        Finds the average skin color for each of the 3 color channels and returns the average as a tuple
        """

        # Filter out the blue colors
        self.filtered_source_skin[(self.filtered_source_skin[:,:,2]>self.filtered_source_skin[:,:,0]) & (self.filtered_source_skin[:,:,2]>self.filtered_source_skin[:,:,1])] = 0
        
        
        fig,axs = plt.subplots(1,1,figsize=(9,9))
        axs.imshow(self.filtered_source_skin)
        plt.show()
        plt.savefig("filtered_skin.png")

        #Calculate the area of the skin
        h,b = self.mask_source.shape
        total_pixel = h*b
        total_pixel_skin = total_pixel - np.sum(self.mask_source)
      
        # Finds the average skin color for each channel
        avg_r = np.sum(self.filtered_source_skin[:,:,0])/total_pixel_skin
        avg_g = np.sum(self.filtered_source_skin[:,:,1])/total_pixel_skin
        avg_b = np.sum(self.filtered_source_skin[:,:,2])/total_pixel_skin

        return (avg_r,avg_g,avg_b)

        

    def get_color_feature(self):
        """
        Returns the color features of the lesion using SLIC

        Uses imports skimage.segmentention.slic and skimage.segmentention.mark_boundaries
        to segment the image and mark the boundaries of the segments

        Then we find the average of each of the segments
        """
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3184884/ color extraction idea
        # https://biomedpharmajournal.org/vol12no1/melanoma-detection-in-dermoscopic-images-using-color-features/ another idea
        # 

        filtered_img = self.filtered_img
        # Segments the image using SLIC
        segments_slic = slic(filtered_img, n_segments=10, compactness=3, sigma=3,start_label=1)
        feat_im = feature.multiscale_basic_features(
            filtered_img,
            channel_axis=2,
            intensity = False, 
            edges = True, 
            texture = True,
            sigma_min = 3,
            sigma_max = 3
            )
        fig,axs = plt.subplots(3,3,figsize=(10,10))

        for i,ax in enumerate(axs.ravel()):
            ax.imshow(feat_im[:,:,i],cmap="gray")

        plt.show()
        plt.savefig("feature.png")
        print("hej")
        #NOT FINISHED

    def __str__(self) -> str:
        return f'{self.path}'

def main():
    #Circular lesion
 
    
    lesion = Lesion("PAT_860_1641_998")
    np.savetxt("before.txt",lesion.mask,fmt="%d")
    lesion.resize_center(buffer=10)
    np.savetxt("mask.txt",lesion.mask,fmt="%d")
    print("Circle lesion \n")
    print("Asymmetry: ",lesion.get_asymmetry_feature())
    print("Asymmetry 2: ", lesion.get_rotation_asymmetry())
    print("Compactness: ",lesion.get_compactness())
    print("Image: ", lesion.image)

    #Oval lesion
    print("\nOval lesion \n")

    lesion = Lesion("PAT_1257_887_828")
    lesion.resize_center(buffer=10)
    print("Asymmetry: ",lesion.get_asymmetry_feature())
    print("Asymmetry 2: ", lesion.get_rotation_asymmetry())
    print("Compactness: ",lesion.get_compactness())


    print("\nColor feature \n")
  
    lesion = Lesion("PAT_599_1140_399")
    lesion.resize_center()
    lesion.apply_mask_to_img()
    print(lesion.average_skin_color())
    
    


# TESTING FUNCTIONS DELETE AFTER

def perfect_circle():
    #Function that makes perfect circle as np.array
    #Returns the circle as np.array
    circle = np.zeros((1000,1000),dtype=int64)
    for i in range(1000):
        for j in range(1000):
            if (i-50)**2+(j-50)**2 <= 50**2:
                circle[i,j] = 1
    return circle.astype(int64)

def perfect_box():
    #Function that makes perfect box as np.array
    #Returns the box as np.array with 1's 
    box = np.zeros((100,100),dtype=int64)
    for i in range(1,98):
        for j in range(1,98):
            box[i,j] = 1
    return box
    
  
if __name__ == '__main__':
    main()
    