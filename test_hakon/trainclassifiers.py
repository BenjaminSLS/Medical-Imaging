import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import math
# Default packages for the minimum example
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score  # example for measuring performance
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import matplotlib.pyplot as plt
import statistics
import pickle  # for saving/loading trained classifiers

file_features = '../features/features.csv'
features_names = ["area", "perimeter", "compactness",
                  "rotation_asymmetry", "asymmetry", "hue_sd", "sat_sd", "val_sd"]
df_features = pd.read_csv(file_features, sep=',',)
X = df_features[features_names]
y = df_features["diagnostic"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

accuracy_scores = []
for k in range(1, 100, 2):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier = classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    scores = cross_val_score(classifier, X, y.ravel(),
                             cv=5)
    accuracy_scores.append(
        (accuracy_score(y_test, y_pred)*100, statistics.mean(scores)*100, k))

# Sort accuracy scores in descending order

accuracy_scores_sorted = sorted(accuracy_scores, key=lambda x: x[1], reverse=True)
max_accuracy = accuracy_scores_sorted[0]

# Best features
selector = SelectKBest(k="all").fit(X_train, y_train)

scores = selector.scores_
features = selector.feature_names_in_
threshold = (statistics.mean(scores)/2)
mask = scores > threshold
reduced_features = features[mask]


#Train reduced model and save it.
classifier = KNeighborsClassifier(n_neighbors=max_accuracy[2])
classifier = classifier.fit(X_train[reduced_features], y_train)
y_pred = classifier.predict(X_test[reduced_features])
scores = cross_val_score(classifier, X[reduced_features], y.ravel())
print("Accuracy score with reduced features: ", accuracy_score(y_test, y_pred))
print("Cross validation score with reduced features: ", statistics.mean(scores))
filename = "../classifier/groupXY_classifier.sav"
pickle.dump(classifier, open(filename, 'wb'))

#Make plot showing best K
plt.plot([x[2] for x in accuracy_scores], [x[1] for x in accuracy_scores])
plt.plot(max_accuracy[2], max_accuracy[1], 'ro')
plt.annotate(f'Maximum Test Accuracy\n---------------------------------\n            K = {max_accuracy[2]}\n     Accuracy: {round(max_accuracy[1],2)}%', 
             xy=(max_accuracy[2], max_accuracy[1]), xytext=(max_accuracy[2]+15, max_accuracy[1]+2), fontsize=12, arrowprops=dict(facecolor="black"))
plt.ylabel("Cross-Validated Accuracy Score")
plt.xlabel("K-Nearest Neighbors")
plt.title("Accuracy Score For Different K-Nearest Neighbors")
plt.grid()
plt.ylim(60,75)
plt.savefig("../plots/knn_plot.png", dpi=400)

#Confusion matrix
