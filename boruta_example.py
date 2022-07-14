import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import functionsBoruta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sklearn.model_selection as model_selection
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, cross_val_predict

pastasName = ["Breast Cancer Wisconsin",
              "Bupa", "Diabetes", "HCC",
              "Hepatitis", "Parkinsons (LPD)",
              "Planning Relax", "SaHeart",
              "SPECTF Heart", "Statlog (Heart)"]

saidaTeste = {}
pastasName = ["Breast Cancer Wisconsin"]

for pasta in pastasName:

    # i = 5

    # print("Dataset Name: ", pasta)
    dataTrain = pd.read_csv(
        'data/' + str(pasta) + '/train.csv', header=None).values

    dataTest = pd.read_csv(
        'data/' + str(pasta) + '/test.csv', header=None).values

    nameFeatures = dataTrain[0, :]
    Xtrain = dataTrain[1:, :-1]
    Xtest = dataTest[1:, :-1]
    Ytrain = dataTrain[1:, -1:]
    Ytest = dataTest[1:, -1:]

    X_filtered_Train = functionsBoruta.filterBoruta(
        Xtrain, Ytrain.ravel(), printed=False)
    saidaTeste[pasta] = X_filtered_Train
    # print("X_filtered: ", X_filtered_Train)
    # print("X_filtered: ", X_filtered_Train['Accepted'])

    # print("_"*100)
    '''
    clf = RandomForestClassifier()  # Initialize with whatever parameters you want to
    cross_validation_result = np.mean(
        cross_val_score(clf, Xtrain, Ytrain.ravel(), cv=10))
    # 10-Fold Cross validation

    print("_"*100)
    print("Dataset Name: ", pasta)
    print(cross_validation_result)
    print("_"*100)
    '''
    # print(Ytrain.ravel())
    '''

    param_grid = {
        'n_estimators': [5, 10, 15, 20],
        'max_depth': [2, 5, 7, 9]
    }

    clf = RandomForestClassifier()  # Initialize with whatever parameters you want to
    grid_clf = GridSearchCV(clf, param_grid, scoring='accuracy', cv=10)
    result = grid_clf.fit(Xtrain, Ytrain.ravel())
    best_model = result.best_estimator_



    print("_"*100)
    print("Dataset Name: ", pasta)
    print("grid_clf.best_estimator_", grid_clf.best_estimator_)
    print("grid_clf.best_params_", grid_clf.best_params_)
    # print("grid_clf.cv_results_", grid_clf.cv_results_)

    # Initialize with whatever parameters you want to

    clf = RandomForestClassifier(
        max_depth=grid_clf.best_params_['max_depth'],
        n_estimators=grid_clf.best_params_['n_estimators'])
    cross_validation_result_grid = np.mean(
        cross_val_score(clf, Xtrain, Ytrain.ravel(), cv=10))
    print(cross_validation_result_grid)

    clf = RandomForestClassifier()
    cross_validation_result = np.mean(
        cross_val_score(clf, Xtrain, Ytrain.ravel(), cv=10))
    print(cross_validation_result)
    '''
############################################################

    clf = RandomForestClassifier(random_state=1)
    cross_validation_result_grid = cross_val_score(
        clf, Xtrain, Ytrain.ravel(), cv=10)
    acc_cross = cross_validation_result_grid.mean()

    # cross_validation_result_grid = cross_val_score(
    #    clf, Xtrain, Ytrain.ravel(), scoring="f1", cv=10)
    # f1_cross = cross_validation_result_grid.mean()

    print("_"*100)
    print("Dataset Name: ", pasta)
    print("acc_cross", acc_cross)
    # print("grid_clf.cv_results_", grid_clf.cv_results_)
    '''

    scoring1 = {'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1_score': make_scorer(f1_score)}
    scoring = ['accuracy',
               'precision',
               'recall',
               'f1_score']
    kfold = model_selection.KFold(n_splits=10, random_state=1, shuffle=True)
    model = RandomForestClassifier(n_estimators=50)
    results = model_selection.cross_validate(estimator=model,
                                             X=Xtrain,
                                             y=Ytrain.ravel(),
                                             cv=kfold,
                                             scoring=scoring)
    print("RESULTS: ", results)

    '''

# print("saidaTeste: ", saidaTeste)
# print("saidaTeste: ", saidaTeste['Statlog (Heart)']['Accepted'])

# print(X_filtered_Train)
    model = RandomForestClassifier(n_estimators=50)

    scores = model_selection.cross_validate(model,
                                            Xtrain,
                                            Ytrain.ravel(),
                                            cv=3,
                                            scoring=(
                                                'accuracy', 'r2'),
                                            return_train_score=True)

    print("Print scores", scores)
