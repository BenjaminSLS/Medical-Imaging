import pickle
from matplotlib import pyplot as plt
import pandas as pd
from extractfeatures import extract_features
import numpy as np


def classify(img, mask):

    # Resize the image etc, if you did that during training

    # Extract features (the same ones that you used for training)
    X = extract_features(img, mask)

    features_index = open("features.txt").readlines()[0].split(" ")
    features_names = ["area", "perimeter", "compactness",
                      "rotation_asymmetry", "asymmetry", "hue_sd", "sat_sd", "val_sd"]
    temp = list(X)
    features = {}
    for i in features_index:
        features[features_names[int(i)]] = temp[int(i)]

    X_test = pd.DataFrame.from_dict(features, orient='index')
    X = X_test.transpose()

    # Load the trained classifier
    classifier = pickle.load(open('groupXY_classifier.sav', 'rb'))

    # Use it on this example to predict the label AND posterior probability
    pred_prob = classifier.predict_proba(X)

    # print('predicted label is ', pred_label)
    # print('predicted probability is ', pred_prob)
    return round(pred_prob[0][1], 2)


# The TAs will call the function above in a loop, for external test images/masks
