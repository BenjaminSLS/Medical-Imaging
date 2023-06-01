import os
import pandas as pd
from lesion import Lesion
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from multiprocessing.pool import ThreadPool as Pool
import concurrent.futures
import pickle


def prepare_data_wrapper(class_instance):
    result = class_instance.prepare_data()
    del class_instance
    return result


def load_lesions(max_count=10000):
    """
    Loads all the lesions from the given path and returns a list of lesions
    """
    df = pd.read_csv("../data/metadata.csv")

    lesions = []

    counter = 0
    files = os.listdir("../segmentation/masks")
    max_length = len(files)
    for file in files:
        patient_id = file.split("_mask")[0]
        metadata = df.loc[df["img_id"] == patient_id+".png"]

        lesion = Lesion(patient_id, metadata)
        lesions.append(lesion)
        counter += 1
        if counter == max_count or counter == max_length - 1:
            return lesions


def make_dataframe(lesions):
    full_df = pd.DataFrame()
    cancer_series = pd.Series([], dtype=bool)

    result_array = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(prepare_data_wrapper, class_instance)
                   for class_instance in lesions]
        for future in concurrent.futures.as_completed(futures):
            result_array.append(future.result())

    for row in result_array:
        full_df = pd.concat([full_df, row[0]])
        cancer_series = pd.concat([cancer_series, pd.Series(row[1])])

    X = full_df
    X.replace({"True": 1, "False": 0, "UNK": 0}, inplace=True)
    y = cancer_series

    ohc_features = ["gender", "region",
                    "background_father", "background_mother"]
    drop_features = ["patient_id", "lesion_id", "img_id", "biopsed", "pesticide", "skin_cancer_history",
                     "smoke", "drink", "has_piped_water", "cancer_history", "has_sewage_system", "fitspatrick", "diameter_1", "diameter_2", "diagnostic"]

    X = ohc(full_df, ohc_features, drop_features)
    X.to_csv("features.csv")

    return X, y


def ohc(dataframe, ohc_features=[], drop_features=[]):
    """
    One hot encodes the dataframe and returns the encoded dataframe
    Specificy which columns to one hot encode and which to drop with lists.
    """
    print("One hot encoding...")
    drop_features.extend(ohc_features)

    dummies = pd.get_dummies(dataframe, columns=ohc_features, dummy_na=True)
    dummies = dummies.drop(
        ["compactness", "rotation-asymmetry", "asymmetry", "hue-sd"], axis=1)

    dataframe = pd.concat([dataframe, dummies], axis=1)
    dataframe = dataframe.drop(drop_features, axis=1)

    dataframe.to_csv("ohc.csv")

    # Make sure errors don't occur for training.
    for column in dataframe.columns:
        if "_nan" in column:
            dataframe = dataframe.drop(column, axis=1)
        elif "UNK" in column:
            dataframe = dataframe.drop(column, axis=1)
        elif column[-1] == "0":
            dataframe = dataframe.drop(column, axis=1)
    dataframe = dataframe.iloc[:, 1:]

    return dataframe


def train_test_validate_split(X, y):
    """
    Splits the data into train, test and validation sets with a 80/10/10 split.
    """
    print("Splitting data...")

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    # 50/50 split of the 20% of the test data
    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, shuffle=False)

    # return X_train, y_train, X_test, y_test, X_test_train, y_test_train # Delete and uncomment below
    return X_train, y_train, X_test_train, y_test_train, X_test_test, y_test_test


def train_model(X_train, y_train, X_test, y_test):
    """
    Trains the model using the training data and returns the model.
    """

    # Build a k-NN classifier and find the optimal number of neighbors

    acurracy_scores = []
    index_list = []
    for n in range(1, 30, 2):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acurracy_scores.append(accuracy_score(y_test, y_pred)*100)
        index_list.append(n)
        print(f"Accuracy score with n= {n}: ",
              accuracy_score(y_test, y_pred)*100)

    max_accuraccy = max(acurracy_scores)
    max_accuraccy_index = index_list[acurracy_scores.index(max_accuraccy)]

    plt.plot(range(1, 30, 2), acurracy_scores)

    # Add line and dot for max accuracy
    plt.plot(max_accuraccy_index, max_accuraccy, 'ro')
    # line = [max_accuraccy,max_accuraccy-(10+(max_accuraccy % 10))]
    # line_index = [max_accuraccy_index+2 for _ in range(len(line))]
    # plt.plot(line_index,line,linestyle="dashed",color="black")
    # plt.text(max_accuraccy_index+2,line_index[1]-5,s="Maximum test accurraccy",fontsize=20)
    plt.annotate(f'Maximum test accuracy\nK = {max_accuraccy_index}\n Accuracy: {round(max_accuraccy,2)} ', xy=(
        max_accuraccy_index, max_accuraccy), xytext=(max_accuraccy_index+15, max_accuraccy-5), fontsize=16, arrowprops=dict(facecolor="black"))
    plt.ylabel("Accuracy score")
    plt.xlabel("K-nearest neighbors")
    plt.title("Accuracy score for different K-nearest neighbors")
    plt.grid()
    plt.savefig("knn.png", dpi=400)

    return max_accuraccy_index


def feature_selection(X_train, y_train):
    """"
    Selects the k best features using mutual information and returns the scores and the selector.
    """
    n = X_train.shape[1]
    selector = SelectKBest(k=n)
    selector.fit(X_train, y_train)
    p_values = selector.pvalues_
    features = selector.feature_names_in_
    print("p_values", p_values)
    features_delete = []
    for f, p in zip(features, p_values):
        if p < 0.05:
            features_delete.append(f)

    return features_delete


def knn_plot(X_train, y_train, X_test, y_test, k):
    """
    Function to make a knn plot
    Saves as a .png file in the main folder.
    """
    print("Making knn plot... with k =", k)
    # Create a KNN classifier object
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Define the classes
    classes = np.unique(y_train)

    # Predict the labels for the test data
    y_pred = knn.predict(X_test)
    # y_pred_proba = knn.predict_proba(X_test)
    # print("probabilities of pictures not being cancerous:",y_pred_proba)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    print("True negative:", tn, "\nFalse positive:", fp,
          "\nFalse negative:", fn, "\nTrue positive:", tp)
    # Plot the confusion matrix
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    plt.savefig("confusionmatrix.png")
    return None


def reduced_model(X_train, y_train, X_test, y_test, k):
    print("\n"*5, "Reduced model:", "\n"*5)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    filename = "groupXY_classifier.sav"
    pickle.dump(knn, open(filename, 'wb'))
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy score: ", accuracy)

    return None


if __name__ == "__main__":
    lesions = load_lesions()
    X, y = make_dataframe(lesions)

    # All data
    X_train, y_train, X_test_train, y_test_train, X_test_test, y_test_test = train_test_validate_split(
        X, y)
    data = [X_train, y_train, X_test_train,
            y_test_train, X_test_test, y_test_test]

    k = train_model(data[0], data[1], data[2], data[3])
    features_delete = feature_selection(X_train, y_train)
    print(features_delete)
    data[0] = data[0].drop(features_delete, axis=1)
    data[2] = data[2].drop(features_delete, axis=1)
    data[4] = data[4].drop(features_delete, axis=1)

    knn_plot(data[0], data[1], data[4], data[5], k)
    reduced_model(data[0], data[1], data[4], data[5], k)
