from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import functionsBoruta
from sklearn.preprocessing import LabelEncoder
import numpy as np


pastasName = ["Breast Cancer Wisconsin",
              "Bupa", "Diabetes", "HCC",
              "Hepatitis", "Parkinsons (LPD)",
              "Planning Relax", "SaHeart",
              "SPECTF Heart", "Statlog (Heart)"]

saidaTeste = {}

for pasta in pastasName:

    #i = 5

    #print("Dataset Name: ", pasta)
    dataTrain = pd.read_csv(
        'data/' + str(pasta) + '/train.csv', header=None).values

    dataTest = pd.read_csv(
        'data/' + str(pasta) + '/test.csv', header=None).values

    nameFeatures = dataTrain[0, :]
    Xtrain = dataTrain[1:, :-1]
    Xtest = dataTest[1:, :-1]
    Ytrain = dataTrain[1:, -1:]
    Ytest = dataTest[1:, -1:]

    X = np.vstack([Xtrain, Xtest])
    Y = np.vstack([Ytrain, Ytest])

    X_filtered_Train = functionsBoruta.filterBoruta(
        X, Y, printed=False)
    saidaTeste[pasta] = X_filtered_Train
    print("X_filtered: ", X_filtered_Train)

#print("saidaTeste: ", saidaTeste)
#print("saidaTeste: ", saidaTeste['Statlog (Heart)']['Accepted'])


# print(X_filtered_Train)
