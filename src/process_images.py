import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from extractfeatures import extract_features
np.set_printoptions(suppress=True)

file_data = "../data/metadata.csv"
path_image = "../data/images/images/"

file_features = "../features/features.csv"

df = pd.read_csv(file_data)


feature_names = ["diagnostic", "area", "perimeter", "compactness",
                 "rotation_asymmetry", "asymmetry", "hue_sd", "sat_sd", "val_sd"]
num_features = len(feature_names)

files = os.listdir("../segmentation/masks")
num_files = len(files)

features = np.zeros([num_files, num_features])

counter = 0
for file in files:
    if counter < 10:
        # if file != "PAT_101_1041_898_mask.npy":
        #     continue
        # if counter < 100000000:
        patient_id = file.split("_mask")[0]
        label = df.loc[df["img_id"] == patient_id+".png"]["diagnostic"]
        label = label.values[0]

        dia: int = 0
        if label in ["BCC", "MEL", "SCC"]:
            dia = 1

        im = plt.imread(path_image + patient_id + ".png")
        mask = np.load("../segmentation/masks/" + file)

        x = extract_features(im, mask)
        dia_arr = np.array(dia)
        new_x = np.zeros(len(x)+1)
        for i in range(len(x)+1):
            if i == 0:
                new_x[i] = dia
            else:
                new_x[i] = x[i-1]
        x = new_x
        list_x = np.ndarray.tolist(x)
        list_x.append(patient_id)

        # if x[1] == np.inf:
        #     print(patient_id,x)
        # print(list_x)
        features[counter, :] = x
        counter += 1
        print(counter)
df_features = pd.DataFrame(features, columns=feature_names)
df_features.to_csv(file_features, index=False, sep=",")
print(df_features)
