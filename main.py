import functions
import pandas as pd

pastasName = ["Breast Cancer Wisconsin",
              "Bupa", "Diabetes", "HCC",
              "Hepatitis", "Parkinsons (LPD)",
              "Planning Relax", "SaHeart",
              "SPECTF Heart", "Statlog (Heart)"]

for pasta in pastasName:

    nameFeatures, Xtrain, Xtest, Ytrain, Ytest = functions.read_arquivo(
        pasta)

    #print("X: ", Xtrain.astype(float))
