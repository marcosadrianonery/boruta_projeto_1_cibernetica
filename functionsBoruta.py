import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import numpy as np
import functions
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def filterBoruta(X, y, printed):

    rf = RandomForestClassifier(n_jobs=-1, max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto',
                             random_state=1, max_iter=100)

    feat_selector.fit(functions.retiraItens(X), y)

    support = feat_selector.support_
    accepted = feat_selector.support_
    undecided = feat_selector.support_weak_

    if printed == True:
        print('support: ', support)
        print('Accepted features:', accepted)
        print('Undecided features', undecided)

    ranking = feat_selector.ranking_
    X_filtered = feat_selector.transform(X)

    if printed == True:
        print("_________________________________________________")
        print("_________________________________________________")
        print(X_filtered)
        print(feat_selector.n_features_)
        print("_________________________________________________")
        print("_________________________________________________")

    return {"Support": support, "Accepted": accepted, "Undecided": undecided, "X_filtered": X_filtered, "Ranking": ranking}
