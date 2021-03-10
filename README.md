# Simulated-Hand-Gestures

# data
contains the 4 data sets for the 4 hand gestures from https://www.kaggle.com/kyr7plus/emg-4?select=0.csv

# 64 Variable Data Analysis
**Hand Gestures_0_M** contains the pairwise analysis for type 0 and type M

**svmfit0M** is the trained radial kernel svm on the PCA preprocessed datasets in **Hand Gestures_0_M**

**4 Type Classification** uses the same test set in all the **Hand Gestures_0_M files** and runs the 6 pairwise SVM classifiers, assigning the class to be the one that was predicted the most in the 6 classifiers. (Extra introduction of tie breaking classifier in the event of ties between Types 1, 2 & 3)

# 8 Sensors Data Analysis

**SVM Cross Validations** contains the fine tuning of the 6 pairwise radial kernel svm classifier plus 1 multi class radial kernel svm classifier for tie breaking. (Requires > 24 Hrs to train and run full script)

**8 Sensors Data Analysis** runs the 6 pairwise SVM classifiers, assigning the class to be the one that was predicted the most in the 6 classifiers. (Extra introduction of tie breaking classifier in the event of ties between Types 1, 2 & 3)
