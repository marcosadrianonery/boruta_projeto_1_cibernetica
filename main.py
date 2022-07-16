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

pastasName = ["Breast Cancer Wisconsin",
              "Bupa", "Diabetes", "HCC",
              "Hepatitis", "Parkinsons (LPD)",
              "SaHeart", "Planning Relax",
              "SPECTF Heart", "Statlog (Heart)"]


#pastasName = ["Planning Relax"]


saidaTeste = {}

for pasta in pastasName:

    n_estimators = 20

    ##############################################################################
    #   DATABASE PART
    ##############################################################################
    print("#"*100)
    print("Dataset Name: ", pasta)
    print("-"*100)

    nameFeatures, Xtrain, Xtest, Ytrain, Ytest = functions.read_arquivo(pasta)

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

    model = RandomForestClassifier(n_estimators=n_estimators)

    metrics = {'precision': 'precision',
               'recall': 'recall',
               'accuracy': 'accuracy',
               'f1': 'f1',
               'roc_auc': 'roc_auc',
               'AUPR': make_scorer(average_precision_score),
               'neg_root_mean_squared_error': make_scorer(mean_squared_error, squared=False)
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

    metricas_cross = [
        scores['test_accuracy'].mean(),
        scores['test_f1'].mean(),
        scores['test_precision'].mean(),
        scores['test_recall'].mean(),
        scores['test_roc_auc'].mean(),
        scores['test_AUPR'].mean(),
        scores['test_neg_root_mean_squared_error'].mean()]

    names = ["Acurácia", "F1",
             "Precision", "Recall",
             "ROC_AUC", "AUPR", "RMSE"]

    #   ----------------------------------------------------------------------------
    scores = cross_validate(model, functions.filterX(X),
                            functions.filterY(Y), cv=10, scoring=metrics)

    metricas_cross_sem_boruta = [
        scores['test_accuracy'].mean(),
        scores['test_f1'].mean(),
        scores['test_precision'].mean(),
        scores['test_recall'].mean(),
        scores['test_roc_auc'].mean(),
        scores['test_AUPR'].mean(),
        scores['test_neg_root_mean_squared_error'].mean()]

    ##############################################################################
    #   TEST PART [Xtrain, Xtest, Ytrain, Ytest]
    ##############################################################################

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_train_filtred, Y_train_filtred, test_size=0.3, shuffle=True, random_state=32)

    model_test = RandomForestClassifier(n_estimators=n_estimators)

    model_test.fit(X_treino, y_treino)

    y_predict = model_test.predict(X_teste)
    relatorio = classification_report(
        y_teste, y_predict)

    metricas_predict_boruta = [
        accuracy_score(y_teste, y_predict),
        f1_score(y_teste, y_predict),
        precision_score(y_teste, y_predict),
        recall_score(y_teste, y_predict),
        roc_auc_score(y_teste, y_predict),
        average_precision_score(y_teste, y_predict),
        mean_squared_error(y_teste, y_predict, squared=False)]
    names_boruta = ["Acurácia", "F1",
                    "Precision", "Recall",
                    "ROC_AUC", "AUPR", "RMSE"]

    print("Acurácia de teste: ", metricas_predict_boruta[0])
    print("f1_score de teste: ", metricas_predict_boruta[1])
    print("precision_score de teste: ", metricas_predict_boruta[2])
    print("recall_score de teste: ", metricas_predict_boruta[3])
    print("roc_auc_score de teste: ", metricas_predict_boruta[4])
    print("precision_recall de teste: ", metricas_predict_boruta[5]),
    print("RMSE de teste: ", metricas_predict_boruta[6])
    print("-"*100)

    #   -----------------------------------------------------------------------

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        functions.filterX(X), functions.filterY(Y), test_size=0.3, shuffle=True, random_state=32)

    model_test = RandomForestClassifier(n_estimators=n_estimators)

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
        average_precision_score(y_teste, y_predict),
        mean_squared_error(y_teste, y_predict, squared=False)]
    names = ["Acurácia", "F1",
             "Precision", "Recall",
             "ROC_AUC", "AUPR", "RMSE"]

    print("Acurácia de teste: ", metricas_predict[0])
    print("f1_score de teste: ", metricas_predict[1])
    print("precision_score de teste: ", metricas_predict[2])
    print("recall_score de teste: ", metricas_predict[3])
    print("roc_auc_score de teste: ", metricas_predict[4])
    print("precision_recall de teste: ", metricas_predict[5]),
    print("RMSE de teste: ", metricas_predict[6])
    print("-"*100)

    #################################################################
    #fig, ax = plt.subplots(2, 1)

    mat_conf = confusion_matrix(y_teste, y_predict, labels=model_test.classes_)
    confusionMatrix = ConfusionMatrixDisplay(
        confusion_matrix=mat_conf, display_labels=model_test.classes_)
    print("Matriz de confusão:")
    print(mat_conf)

    confusionMatrix.plot()
    plt.savefig('imagens/' + pasta + '_confusionMatrix.png')

    #################################################################

    width = 0.75

    fig_plote = [
        metricas_cross,
        metricas_cross_sem_boruta,
        metricas_predict_boruta,
        metricas_predict]

    fig_names = [
        "Cross Validation com uso do boruta",
        "Cross Validation sem uso do boruta",
        "Predição com uso do boruta",
        "Predição sem uso do boruta"]

    cores = ['black', '#EF863C', '#8D19A7', '#89DBDC']

    df = pd.DataFrame(
        {fig_names[0]: fig_plote[0], fig_names[1]: fig_plote[1],
         fig_names[2]: fig_plote[2], fig_names[3]: fig_plote[3]},
        index=names_boruta)

    df.plot.bar(color=cores, width=width, figsize=(20, 10), fontsize=18)
    # plt.legend(fontsize=20)
    #plt.xticks(pos, city)
    plt.xticks(rotation=0)

    #plt.xlabel('Métricas', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.legend(bbox_to_anchor=(0, -0.135, 1., 1.), loc='lower center',
               ncol=2, mode="expand", borderaxespad=0., fontsize=16)
    plt.title('Resultado para o database: ' + pasta, fontsize=24)
    plt.autoscale(axis='x', tight=True)

    plt.savefig('imagens/' + pasta + '_results.png')

    #################################################################
    # ARQUIVO TEXTO
    #################################################################
    names = [" Acuracia:  ", "F1:        ",
             "Precision: ", "Recall:    ",
             "ROC_AUC:   ", "AUPR:      ",
             "RMSE:      "]

    arquivo = open('resultados/' + pasta + '.txt', 'w')

    for index, metrica in enumerate(fig_plote):
        write = np.block([names, metrica])
        arquivo.write("#"*70 + "\n")
        arquivo.write("#" + "   Database " + pasta + ": \n")
        arquivo.write("-"*70 + "\n")
        arquivo.write("#" + "   tipo de dados: " + fig_names[index] + ": \n")
        arquivo.write("#"*70 + "\n")
        arquivo.write("\n")
        string_out = re.sub("\[|\]|\,", "", str(write.T))
        string_out = re.sub("\'", "", string_out)
        arquivo.write(string_out + "\n\n")
        arquivo.write("#"*70 + "\n")
        arquivo.write("\n")

    # arquivo.write(metricas_cross)

    arquivo.close()

# plt.show()
# ax.show()
