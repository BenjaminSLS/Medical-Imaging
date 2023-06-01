from extractfeatures import extract_features
import matplotlib.pyplot as plt
import numpy as np


img_ids = ["PAT_1335_1181_21", "PAT_1364_1246_420", "PAT_621_1183_56","PAT_21_982_266", "PAT_1451_1562_545"]
for image in img_ids:
    img = plt.imread("../data/images/images/" + image +".png")
    mask = np.load("../segmentation/masks/" + image + "_mask.npy")
    features_= extract_features(img,mask)
    print(image, features_)
