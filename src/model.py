import os

import pandas as pd
from lesion import Lesion
from sklearn.model_selection import train_test_split


lesions = []


# Questions:
# 1. Correlation matrix
# 2. Multiple lesions in one image
# 

def loading_bar(counter, total, length=50):
    progress = int(round(length * counter / float(total)))
    percent = int(round(100.0 * counter / float(total)))
    bar = '=' * progress + '-' * (length - progress)
    print(f'[{bar}] {percent}%\r', end='',flush=True)
    if counter == total:
        print('\n')

def load_lesions():
    """
    Loads all the lesions from the given path and returns a list of lesions
    """
    df = pd.read_csv("../data/metadata.csv")

    
    #blobs = []

    counter = 0
    total = len(os.listdir("../segmentation/masks"))
    for file in os.listdir("../segmentation/masks"):

        # mask = np.load("../segmentation/masks/"+file)
        # mask[mask>0] = 1
        #labeled,ncomponents = label(mask,structure)
        #print("Components",ncomponents) 
        #labeled_image, count = skimage.measure.label(mask, return_num=True)
        #print("File: "+ file, "Count: ",count)
        # if count > 1:
        #     blobs.append(file)
        #     pass
        if counter % 10 == 0:
            loading_bar(counter,total)
 
        patient_id = file.split("_mask")[0]
        metadata = df.loc[df["img_id"] == patient_id+".png"]
        #print(patient_id,metadata)
        lesion = Lesion(patient_id,metadata)
        lesion.resize_center()
        lesion.apply_mask_to_img()
        lesions.append(lesion)
        
        counter += 1

        if counter == 100:
            return lesions


        
        

    

    #Code to remove blobs   
    # for blob in blobs:
    #     os.remove("../segmentation/masks/"+blob) 
    #     print(blob)  


def prepare_data():
    full_df = pd.DataFrame()
    cancer_series = pd.Series([])

    for idx,lesion in enumerate(lesions):
        print(idx)
        data = lesion.prepare_data()
        full_df = pd.concat([full_df,data[0]])
        cancer_series = pd.concat([cancer_series,pd.Series([data[1]])])

    print(full_df)
    print(cancer_series)
   
    X = full_df
    y = cancer_series

    return X,y

def train_test_validate_split(X,y):
    """
    Splits the data into train, test and validation sets with a 80/10/10 split.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)
    Xtest_train, Xtest_test, ytest_train, ytest_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1, shuffle=True, stratify=y)


    return X_train, X_test, y_train, y_test, Xtest_train, Xtest_test, ytest_train, ytest_test
    







if __name__ == "__main__":
    print(load_lesions())
    X,y = prepare_data()
    train_test_validate_split(X,y)
