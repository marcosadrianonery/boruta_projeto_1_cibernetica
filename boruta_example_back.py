import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import functionsBoruta
import functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sklearn.model_selection as model_selection
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, mean_squared_error
pastasName = ["Breast Cancer Wisconsin",
              "Bupa", "Diabetes", "HCC",
              "Hepatitis", "Parkinsons (LPD)",
              "Planning Relax", "SaHeart",
              "SPECTF Heart", "Statlog (Heart)"]
pastasName = ["Breast Cancer Wisconsin",
              "Bupa", "Diabetes", "HCC",
              "Hepatitis", "Parkinsons (LPD)",
              "SaHeart",
              "SPECTF Heart", "Statlog (Heart)"]
saidaTeste = {}
#pastasName = ["Bupa"]

for pasta in pastasName:

    # i = 5

    print("Dataset Name: ", pasta)

    nameFeatures, Xtrain, Xtest, Ytrain, Ytest = functions.read_arquivo(pasta)

    print("-"*1000)
    #print("XXXX: ", Xtrain)
    #print("YYYYY: ", Ytrain)

    X_filtered_Train = functionsBoruta.filterBoruta(
        Xtrain, Ytrain.ravel(), printed=False)
    saidaTeste[pasta] = X_filtered_Train

    X_train_filtred = functions.filterX(saidaTeste[pasta]['X_filtered'])

    Y_train_filtred = functions.filterY(Ytrain.ravel())

    print("Saida Teste: ", X_train_filtred)
    print("Saida Teste: ", Y_train_filtred)

############################################################
    """
    clf = RandomForestClassifier(random_state=1)
    cross_validation_result_grid = cross_val_score(
        clf, Xtrain, Ytrain.ravel(), cv=10)
    acc_cross = cross_validation_result_grid.mean()

    print("_"*100)
    print("Dataset Name: ", pasta)
    print("acc_cross", acc_cross)

    model = RandomForestClassifier(n_estimators=50)

    knn_pipeline = Pipeline(steps=[
        ("normalizacao", MinMaxScaler()),
        ("model", model)
    ])

    scores = cross_validate(knn_pipeline, Xtrain,
                            Ytrain.ravel(), cv=5, scoring=("accuracy", "f1_macro"))
    """
    model = RandomForestClassifier(n_estimators=50)

    metrics = {'precision': 'precision_macro',
               'recall': make_scorer(recall_score, average='macro'),
               'accuracy': 'accuracy',
               'f1_macro': 'f1_macro',
               'roc_auc': 'roc_auc',
               # Para classificação binária, mas com rótulos diferentes de 0 ou 1, devemos especificar o valor positivo no arg pos_label (nesse caso=4)
               # Para classificação binária, mas com rótulos diferentes de 0 ou 1, devemos especificar o valor positivo no arg pos_label (nesse caso=4)
               # 'AUPR': make_scorer(average_precision_score),
               'neg_root_mean_squared_error': make_scorer(mean_squared_error)
               }
    scores = cross_validate(model, X_train_filtred,
                            Y_train_filtred, cv=10, scoring=metrics)
    '''
    scores = model_selection.cross_validate(model,
                                            Xtrain,
                                            Ytrain.ravel(),
                                            cv=5,
                                            scoring=(
                                                'accuracy', 'r2'))
    '''
    print("Print scores", scores)
