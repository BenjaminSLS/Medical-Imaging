import os
import pandas as pd
from lesion import Lesion
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import confusion_matrix

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

def load_lesions(max_count=300):
    """
    Loads all the lesions from the given path and returns a list of lesions
    """
    print("Loading lesions...")
    df = pd.read_csv("../data/metadata.csv")

    #blobs = []
    lesions = []
    lesions_not_biopsied = []

    counter = 0

    files = os.listdir("../segmentation/masks")
    for file in files:

        # mask = np.load("../segmentation/masks/"+file)
        # mask[mask>0] = 1
        #labeled,ncomponents = label(mask,structure)
        #print("Components",ncomponents) 
        #labeled_image, count = skimage.measure.label(mask, return_num=True)
        #print("File: "+ file, "Count: ",count)
        # if count > 1:
        #     blobs.append(file)
        #     pass
            #Code to remove blobs   
        # for blob in blobs:
        #     os.remove("../segmentation/masks/"+blob)
        #     print(blob)  

        if counter % 4 == 0:
            loading_bar(counter,max_count)
 
        patient_id = file.split("_mask")[0]
        metadata = df.loc[df["img_id"] == patient_id+".png"]
        #print(patient_id,metadata)
        lesion = Lesion(patient_id,metadata)
        biopsied = metadata["biopsed"].values[0]
        lesion.resize_center()
        lesion.apply_mask_to_img()

        if biopsied:
            lesions.append(lesion)
            lesions_not_biopsied.append(lesion) 
        else:
            lesions_not_biopsied.append(lesion)
   
        counter += 1

        if counter == max_count:
            return lesions,lesions_not_biopsied

def make_dataframe(lesions,lesions_not_biopsied):
    print("Preparing data...")
    loaded_counter = 0 
    full_df = pd.DataFrame()
    cancer_series = pd.Series([])

    for idx,lesion in enumerate(lesions):
        data = lesion.prepare_data()
        full_df = pd.concat([full_df,data[0]])
        cancer_series = pd.concat([cancer_series,pd.Series([data[1]])])
        if idx %2 == 0:
            loading_bar(loaded_counter,len(lesions_not_biopsied)+len(lesions))
        loaded_counter += 1

    X = full_df
    y = cancer_series

    ohc_features = ["has_piped_water","has_sewage_system", 
                "smoke", "drink","pesticide","skin_cancer_history","cancer_history",
                "itch","grew","hurt","changed","bleed","elevation"]
    drop_features = ["patient_id","lesion_id","img_id","gender","region","diagnostic","background_father","background_mother","biopsed","age"]
    X = ohc(full_df,ohc_features,drop_features)
    X.to_csv("ohc.csv")
    
    full_df = pd.DataFrame()
    cancer_series = pd.Series([])

    for idx, lesion in enumerate(lesions_not_biopsied):
        data = lesion.prepare_data()
        full_df = pd.concat([full_df,data[0]])
        cancer_series = pd.concat([cancer_series,pd.Series([data[1]])])
        if idx %2 == 0:
            loading_bar(loaded_counter,len(lesions_not_biopsied)+len(lesions))
        loaded_counter += 1

    X_not_biopsied = full_df
    y_not_biopsied = cancer_series
    ohc_features = ["itch","grew","hurt","changed","bleed","elevation"]
    drop_features = ["patient_id","lesion_id","img_id","gender","region","diagnostic",
                     "background_father","background_mother","biopsed","age","pesticide","skin_cancer_history", 
                "smoke", "drink","has_piped_water","cancer_history","has_sewage_system","fitspatrick","diameter_1","diameter_2"]

    X_not_biopsied = ohc(full_df,ohc_features,drop_features)
    X_not_biopsied.to_csv("ohc_not_biopsied.csv")

    return X,y,X_not_biopsied,y_not_biopsied

def ohc(dataframe,ohc_features=[],drop_features=[]) :
    """
    One hot encodes the dataframe and returns the encoded dataframe
    Specificy which columns to one hot encode and which to drop with lists.
    """
    print("One hot encoding...")
    drop_features.extend(ohc_features)
    
    dummies = pd.get_dummies(dataframe,columns=ohc_features,dummy_na=True)
    #print("Dummy: ",dummies)
    
    dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe = dataframe.drop(drop_features,axis=1)

    for column in dataframe.columns:
        if "_nan" in column:
            dataframe = dataframe.drop(column,axis=1)
        elif "UNK" in column:
            dataframe = dataframe.drop(column,axis=1)
    dataframe = dataframe.iloc[:,1:]
    return dataframe

def train_test_validate_split(X,y):
    """
    Splits the data into train, test and validation sets with a 80/10/10 split.
    """
    print("Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
  
    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, shuffle=True, stratify=y_test)
  
    return X_train, y_train , X_test_train, y_test_train, X_test_test, y_test_test
    
def train_model(X_train,y_train,X_test,y_test):
    """
    Trains the model using the training data and returns the model.
    """


    # Build a k-NN classifier and find the optimal number of neighbors

    acurracy_scores = []
    for n in range(2,50):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)
        acurracy_scores.append(accuracy_score(y_test,y_pred)*100)
        print(f"Accuracy score with n= {n}: ", accuracy_score(y_test,y_pred)*100)
    
    max_accuraccy = max(acurracy_scores)
    max_accuraccy_index = acurracy_scores.index(max_accuraccy)

    plt.plot(range(2,50),acurracy_scores)
    
    #Add line and dot for max accuracy
    plt.plot(max_accuraccy_index+2,max_accuraccy,'ro')
    line = [max_accuraccy,max_accuraccy-(10+(max_accuraccy % 10))]
    line_index = [max_accuraccy_index+2 for _ in range(len(line))]
    plt.plot(line_index,line,linestyle="dashed",color="black")
    #plt.text(max_accuraccy_index+2,line_index[1]-5,s="Maximum test accurraccy",fontsize=20)
    plt.annotate('Maximum test accuracy', xy=(max_accuraccy_index+2, line[1]), xytext=(max_accuraccy_index+2, line[1]-1.5), fontsize=16)
    plt.ylabel("Accuracy score")
    plt.xlabel("K-nearest neighbors")
    plt.title("Accuracy score for different K-nearest neighbors")
    plt.grid()
    plt.savefig("knn.png",dpi=400)
        
    return max_accuraccy_index+2

def feature_selection(X_train, y_train, k):
    """"
    Selects the k best features using mutual information and returns the scores and the selector.
    """    
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    scores = selector.scores_
    features = selector.feature_names_in_
    for result in zip(features,scores):
        print(result)
    return scores, selector

def knn_plot(X_train,y_train,X_test,y_test,k):
    """
    Function to make a knn plot
    Saves as a .png file in the main folder.
    """
    print("Making knn plot...")
    # Create a KNN classifier object
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Define the classes
    classes = np.unique(y_train)

    # Predict the labels for the test data
    y_pred = knn.predict(X_test)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    return None

if __name__ == "__main__":
    lesions,lesions_not_biopsied = load_lesions()
    X,y,X_not_biopsied,y_not_biopsied = make_dataframe(lesions,lesions_not_biopsied)

    print(X)
    print(y)

    #X_train, y_train , X_test_train, y_test_train, X_test_test, y_test_test = train_test_validate_split(X,y)
    #data = [X_train, y_train , X_test_train, y_test_train, X_test_test, y_test_test]

    X_train, y_train , X_test_train, y_test_train, X_test_test, y_test_test = train_test_validate_split(X_not_biopsied,y_not_biopsied)
    non_biopsied_data = [X_train, y_train , X_test_train, y_test_train, X_test_test, y_test_test]

    #k_biopsied = train_model(data[0],data[1],data[2],data[3])
    k_non_biopsied = train_model(non_biopsied_data[0],non_biopsied_data[1],non_biopsied_data[2],non_biopsied_data[3])
    #score,selector = feature_selection(X_train,y_train,5)
    knn_plot(non_biopsied_data[0],non_biopsied_data[1],non_biopsied_data[2],non_biopsied_data[3],k_non_biopsied)
