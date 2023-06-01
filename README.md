# Medical Imaging - Group 5

## How to run the code

If you only want to test the probability of single image/mask being cancerouos or healthy, import the `classify(image,mask)` function from `extractfeatures.py`

### Manual Ratings
The algorithms ratings of the lesions can be found in the `selected_images_features.csv`
And the manual ones in `Annotations_manually.csv`

### Preparition of the images

The images from the three folders should be put into a new folder called `images` in this folder.

### Running the code

If you wish the run the full code, then run the scripts in the following order:

1. `process_images.py`
2. `trainclassifiers.py`

Then after this a single image can be tested in the validation script èvaluate_classifier.py`

## Dependencies

The code was written to work on **python@3.10.4**
The following packages are required to run the code:

-   scipy
-   skimage
-   pandas
-   numpy
-   matplotlib
-   pickle

## Explanation of the code

While working on the project we used the `model.py` and `lesion.py`, to build the foundation of the project. Here we used various techniques to assist in developing the model such as, object oriented programming and multi threading. We then later transfered the code and modified it to fit the current status of our model, described above.
