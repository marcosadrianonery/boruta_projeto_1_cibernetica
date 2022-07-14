import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import functionsBoruta
import functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sklearn.model_selection as model_selection
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pastasName = ["Breast Cancer Wisconsin",
              "Bupa", "Diabetes", "HCC",
              "Hepatitis", "Parkinsons (LPD)",
              "SaHeart", "Planning Relax",
              "SPECTF Heart", "Statlog (Heart)"]

#pastasName = ["Planning Relax"]


saidaTeste = {}

for pasta in pastasName:

    ##############################################################################
    #   DATABASE PART
    ##############################################################################
    print("-"*100)
    print("Dataset Name: ", pasta)
    print("-"*100)

    nameFeatures, Xtrain, Xtest, Ytrain, Ytest = functions.read_arquivo(pasta)

    # print("XXXX: ", Xtrain)
    # print("YYYYY: ", Ytrain)
    X = np.block([[Xtrain], [Xtest]])
    Y = np.block([[Ytrain], [Ytest]])

    X_filtered_Train = functionsBoruta.filterBoruta(
        X, Y.ravel(), printed=False)

    saidaTeste[pasta] = X_filtered_Train

    if np.size(saidaTeste[pasta]['X_filtered']) > 0:
        X_train_filtred = functions.filterX(saidaTeste[pasta]['X_filtered'])
        Y_train_filtred = functions.filterY(Y.ravel())
    else:
        X_train_filtred = functions.filterX(X)
        Y_train_filtred = functions.filterY(Y.ravel())

    ##############################################################################
    #   CROSS VALIDATION PART
    ##############################################################################

    model_normalizado = Pipeline(steps=[
        ("normalizacao", MinMaxScaler()),
        ("RFC", RandomForestClassifier(n_estimators=5))
    ])

    model = RandomForestClassifier(n_estimators=5)

    metrics = {'precision': 'precision',
               'recall': 'recall',
               'accuracy': 'accuracy',
               'f1': 'f1',
               'roc_auc': 'roc_auc',
               'AUPR': make_scorer(average_precision_score),
               'neg_root_mean_squared_error': make_scorer(mean_squared_error)
               }
    scores = cross_validate(model, X_train_filtred,
                            Y_train_filtred, cv=10, scoring=metrics)

    print('Accuracy:', scores['test_accuracy'].mean())
    print('F1:', scores['test_f1'].mean())
    print('Precision:', scores['test_precision'].mean())
    print('Recall:', scores['test_recall'].mean())
    print('ROC-AUC:', scores['test_roc_auc'].mean())
    print('AUPR:', scores['test_AUPR'].mean())
    print('RMSE:', scores['test_neg_root_mean_squared_error'].mean())
    print("-"*100)

    ##############################################################################
    #   TEST PART [Xtrain, Xtest, Ytrain, Ytest]
    ##############################################################################

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_train_filtred, Y_train_filtred, test_size=0.3, shuffle=True, random_state=32)

    print("y_treino", y_treino)
    print("y_teste", y_teste)
    print("Y_train_filtred", Y_train_filtred)

    model_test = RandomForestClassifier(n_estimators=5)

    model_test.fit(X_treino, y_treino)

    y_predict = model_test.predict(X_teste)
    relatorio = classification_report(
        y_teste, y_predict)

    metricas_predict = [
        accuracy_score(y_teste, y_predict),
        f1_score(y_teste, y_predict),
        precision_score(y_teste, y_predict),
        recall_score(y_teste, y_predict),
        roc_auc_score(y_teste, y_predict),
        mean_squared_error(y_teste, y_predict)]
    names = ["Acurácia", "F1",
             "Precision", "Recall",
             "ROC_AUC", "RMSE"]

    # print(relatorio)
    print("Acurácia de teste: ", metricas_predict[0])
    print("f1_score de teste: ", metricas_predict[1])
    print("precision_score de teste: ", metricas_predict[2])
    print("recall_score de teste: ", metricas_predict[3])
    print("roc_auc_score de teste: ", metricas_predict[4])
    print("RMSE de teste: ", metricas_predict[5])

    print("-"*100)

    mat_conf = confusion_matrix(y_teste, y_predict, labels=model_test.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=mat_conf, display_labels=model_test.classes_)
    print("Matriz de confusão:")
    print(mat_conf)
    disp.plot()

    width = 0.15
    fig, ax = plt.subplots()

    plt.bar(names, metricas_predict, color='blue',
            width=width, edgecolor='white', label='Median_Wealth')

    ax.set_ylabel('Valores')
    ax.set_title('Agrupados')
    ax.set_xticks(names)
    # ax.set_xticklabels(metricas_predict)
    ax.legend()
    ax.plot()

plt.show()
ax.show()
