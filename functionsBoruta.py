import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import numpy as np
import functions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def filterBoruta(X, y, printed):
    '''
    load X and y
    NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    X = pd.read_csv('examples/test_X.csv', index_col=0).values
    y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    X = pd.read_csv('examples/test_X.csv', index_col=0).values
    y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    '''
    # y = y.ravel()

    if printed == True:
        print("_________________________________________________")
        print("X: ", X)
        print("_________________________________________________")
        print("y: ", y)
        print("_________________________________________________")
        print("_________________________________________________")

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, max_depth=5)

    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto',
                             random_state=1, max_iter=100)
    # verbose=2, random_state=1, max_iter=100)

    feat_selector.fit(functions.retiraItens(X), y)

    # check selected features - first 5 features are selected
    support = feat_selector.support_
    accepted = feat_selector.support_
    undecided = feat_selector.support_weak_

    if printed == True:
        print('support: ', support)
        print('Accepted features:', accepted)
        print('Undecided features', undecided)

    # check ranking of features
    ranking = feat_selector.ranking_
    # print('Ranking: ', ranking)
    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)

    if printed == True:
        print("_________________________________________________")
        print("_________________________________________________")
        print(X_filtered)
        print(feat_selector.n_features_)
        print("_________________________________________________")
        print("_________________________________________________")

    return {"Support": support, "Accepted": accepted, "Undecided": undecided, "X_filtered": X_filtered}
