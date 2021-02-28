# Simulated-Hand-Gestures

# data
contains the 4 data sets for the 4 hand gestures from https://www.kaggle.com/kyr7plus/emg-4?select=0.csv

# src
**Hand Gestures_0_M** contains the pairwise analysis for type 0 and type M

**svmfit0M** is the trained radial kernel svm on the PCA preprocessed datasets in **Hand Gestures_0_M**

**4 Type Classification** uses the same test set in all the **Hand Gestures_0_M files** and runs the 6 pairwise SVM classifiers, assigning the class to be the one that was predicted the most in the 6 classifiers.
