import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import math
# Default packages for the minimum example
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score  # example for measuring performance
from sklearn.feature_selection import SelectKBest, chi2

import statistics

import pickle  # for saving/loading trained classifiers

file_features = '../features/features.csv'
features_names = ["area", "perimeter", "compactness",
                  "rotation_asymmetry", "asymmetry", "hue_sd", "sat_sd", "val_sd"]
df_features = pd.read_csv(file_features, sep=',',)
print(df_features[features_names])
X = df_features[features_names]
y = df_features["diagnostic"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

accuracy_scores = []
for k in range(1, 30, 2):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier = classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    scores = cross_val_score(classifier, X, y.ravel(),
                             cv=5)
    accuracy_scores.append(
        (accuracy_score(y_test, y_pred), statistics.mean(scores), k))
    # print("Accuracy score with k= ", k, ": ", accuracy_score(y_test, y_pred))

# Sort accuracy scores in descending order
accuracy_scores = sorted(accuracy_scores, key=lambda x: x[0], reverse=True)
max_accuracy = accuracy_scores[0]
print(accuracy_scores)
print(max_accuracy)


# Best features
selector = SelectKBest(k=4).fit(X_train, y_train)
scores = selector.scores_
features = selector.feature_names_in_
print("scores", scores)
print(features)
