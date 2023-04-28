from matplotlib import pyplot as plt
import numpy as np
from numpy import int64, ndarray as NDArray, float64,generic
from typing import Any
from skimage import morphology
from scipy.ndimage import rotate

class Lesion:
    lesion_id : str
    path : str
    mask_source : any
    mask: any
    top : int
    bottom : int
    left : int
    right : int
    
    

    def __init__(self, id: str, mask:any, path: str) -> None:
        self.lesion_id = id
        self.path = path
        self.mask = mask
        self.mask_source = mask
    


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
       
        # Resize the mask
        self.mask = self.mask[top:bottom, left:right]

        return self.mask

    def get_asymmetry_feature(self,mask_in= None):
        """
        Find asymmetric index of mask by folding horizontally and vertically
        """
        mask : any
        if mask_in is None:
            mask =self.mask
        else:
            mask = mask_in


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

        #Rotates the mask by 45 degrees
        rotated_mask = rotate(self.mask,angle=45)
        
        return self.get_asymmetry_feature(rotated_mask)

    def get_compactness(self,mask_in = None):
        """
        Returns the compactness of the lesion
        """
        
        mask : any
        if mask_in is None:
            mask =self.mask
        else:
            mask = mask_in

        
        #Gets the area and perimeter of the lesion using functions _get_area and _get_perimeter
        area = self._get_area()
        perimeter = self._get_perimeter()


        #Calculates the compactness of the lesion
        compactness = perimeter ** 2 / (4 * np.pi * area)
                      
        return compactness
    
    def _get_area(self,mask_in = None) -> int:
        '''
        Returns the area of the mask
        '''
        mask :any
        if mask_in is None:
            mask =  self.mask
        else:
            mask = mask_in
            
        return np.sum(mask)
    
    def _get_perimeter(self, erosion_size : int = 1,mask_in=None):
        """
        Returns the perimeter of the mask
        """
        mask :any
        if mask_in is None:
            mask =  self.mask
        else:
            mask = mask_in
        # Defines a disk brush of size erosion_size
        struct_el = morphology.disk(erosion_size)

        # Erodes the mask with the disk brush
        mask_eroded = morphology.binary_erosion(mask, struct_el)
        
        np.savetxt("mask_eroded.txt",mask_eroded,fmt="%d")

        # Finds the size of the perimeter by subtracting the eroded mask from the original mask
        perimeter = np.sum(mask - mask_eroded)
        print(f'Perimeter {perimeter}')
    
        return perimeter

    def __str__(self) -> str:
        return f'{self.path}'

def main():
    #Circular lesion
    mask = np.load("../segmentation/masks/PAT_860_1641_998_mask.npy")
    mask[mask > 0] = 1
    lesion = Lesion(id="",mask=mask,path="")
    np.savetxt("before.txt",lesion.mask,fmt="%d")
    lesion.resize_center(buffer=10)
    np.savetxt("mask.txt",lesion.mask,fmt="%d")
    print("Circle lesion \n")
    print("Asymmetry: ",lesion.get_asymmetry_feature())
    print("Asymmetry 2: ", lesion.get_rotation_asymmetry())
    print("Compactness: ",lesion.get_compactness())

    #Oval lesion
    print("\nOval lesion \n")
    mask = np.load("../segmentation/masks/PAT_1257_887_828_mask.npy")
    mask[mask > 0] = 1
    lesion = Lesion(id="",mask=mask,path="")
    lesion.resize_center(buffer=10)
    print("Asymmetry: ",lesion.get_asymmetry_feature())
    print("Asymmetry 2: ", lesion.get_rotation_asymmetry())
    print("Compactness: ",lesion.get_compactness())
    
    #Perfect cicle
    print("\nPerfect circle \n")
    mask = perfect_circle()
    mask[mask > 0] = 1
    lesion = Lesion(id="",mask=mask,path="")
    lesion.resize_center(buffer=0)
    print("Asymmetry: ",lesion.get_asymmetry_feature())
    print("Asymmetry 2: ", lesion.get_rotation_asymmetry())
    print("Compactness: ",lesion.get_compactness())
    
    #Perfect box
    print("\nPerfect box \n")
    mask = perfect_box()
    mask[mask > 0] = 1
    lesion = Lesion(id="",mask=mask,path="")
    print("Asymmetry: ",lesion.get_asymmetry_feature())
    print("Asymmetry 2: ", lesion.get_rotation_asymmetry()) # works incorrect here 
    print("Compactness: ",lesion.get_compactness())
    


# TESTING FUNCTIONS

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
    