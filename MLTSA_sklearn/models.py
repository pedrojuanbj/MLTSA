
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

"""

This part of the code contains wrappers to easily train and test the machine learning models, and other intersting 
functions useful for implementing and testing our MLTSA. The functions here are fully implemented using the sklearn suite

"""


def SKL_Train(clf, X, Y):
    """
    Wrapper to train any machine learning model/classifier from the Scikit-Learn suite which uses fit() to train and
    predict() to predict the outcome values.

    :param clf: built model for training, typically model template imported from sklearn
    :type clf: sklearn models class
    :param X: Input data, in shape of (n_samples * n_steps, n_features)
    :param Y: Input labels for the data, in shape of (n_samples * n_steps,)
    :return: trained model, training accuracy, testing accuracy.
    """

    st = time.time()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=1)

    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    train_acc = y_pred_train == y_train
    train_acc = np.mean(train_acc)
    print("Accuracy on Train is", train_acc*100)

    y_pred = clf.predict(X_test)
    test_acc = y_pred == y_test
    test_acc = np.mean(test_acc)
    print("Accuracy on Test set is", test_acc*100)

    print("Trained in ", (time.time() - st), "seconds")

    return clf, train_acc, test_acc
