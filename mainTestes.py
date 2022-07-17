from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, recall_score, make_scorer, precision_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import functionsBoruta
import functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

pastasName = ["HCC"]


# pastasName = ["Planning Relax"]


saidaTeste = {}

bestPercent = {}
bestPercentValue = {}

for pasta in pastasName:

    n_estimators = 100

    ##############################################################################
    #   DATABASE PART
    ##############################################################################
    print("#"*100)
    print("Dataset Name: ", pasta)
    print("-"*100)

    nameFeatures, Xtrain_dataset, Xtest_dataset, Ytrain_dataset, Ytest_dataset = functions.read_arquivo(
        pasta)

    X = np.block([[Xtrain_dataset], [Xtest_dataset]])
    Y = np.block([[Ytrain_dataset], [Ytest_dataset]])

    model = RandomForestClassifier(n_estimators=n_estimators)

    X_filtered_Train = functionsBoruta.filterBoruta(
        X, Y.ravel(), model, n_estimators=n_estimators, perc=100, printed=False)
    saidaTeste[pasta] = X_filtered_Train

#################################################################################
    features = np.array(
        list(enumerate(saidaTeste[pasta]['Accepted'].T)))
    Xtrain_dataset_filter = functions.escolheFeatures(Xtrain_dataset, features)
    Xtest_dataset_filter = functions.escolheFeatures(Xtest_dataset, features)
    # print(features)
