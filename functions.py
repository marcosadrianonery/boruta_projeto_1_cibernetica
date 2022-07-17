import numpy as np
import pandas as pd


def filterY(y):

    min_num = 99999999
    max_num = -9999999
    y_out = []
    for dado in y:

        if float(dado) < min_num:
            min_num = float(dado)
        if float(dado) > max_num:
            max_num = float(dado)

    for dado in y:
        if float(dado) == float(min_num):
            # y_out[pos] = 0
            y_out.append(0)

        if float(dado) == float(max_num):
            # y[pos] = 1
            y_out.append(1)

    return y_out


def filterX(X):

    X_filter = []
    X_linha = []

    for linha in X:
        X_linha = []

        for coluna in linha:
            X_linha.append(float(coluna))

        X_filter.append(X_linha)

    return X_filter


def retiraItens(X):

    PosFeature = []
    colFeature = []
    lineFeature = []
    alterado = []

    for indexLine, lineDado in enumerate(X):
        for indexDado, dado in enumerate(lineDado):
            if dado == "?":
                # print(dado, "Coluna: ", indexDado)
                X[indexLine, indexDado] = 0
                PosFeature.append([indexLine, indexDado])
                colFeature.append(indexDado)
                lineFeature.append(indexLine)
            if dado == "Absent":
                # print(dado, "Coluna: ", indexDado)
                X[indexLine, indexDado] = 0
            if dado == "Present":
                # print(dado, "Coluna: ", indexDado)
                X[indexLine, indexDado] = 1

    mediaValue = []
    for coluna in colFeature:
        mediaValue.append([coluna, np.mean(list(map(float, X[:, coluna])))])

    for line, coluna in PosFeature:

        for colunaMedia, media in mediaValue:
            if coluna == colunaMedia:
                X[line, coluna] = media
                alterado.append([line, coluna, media])

    return X


def normalizarDados(data):

    df_min_max_scaled = []

    for column in data.T:

        max_value = column.max()
        min_value = column.min()

        columnVetor = []

        for value in column:
            if min_value == 0 and max_value == 0:

                columnVetor.append(value/1)

            else:
                columnVetor.append((value - min_value)/(max_value - min_value))

        df_min_max_scaled.append(columnVetor)

    # print(data.T[4])

    # print(df_min_max_scaled[4])
    return np.array(df_min_max_scaled).T


def read_arquivo(nome):

    dataTrain = pd.read_csv(
        'data/' + str(nome) + '/train.csv', header=None).values

    dataTest = pd.read_csv(
        'data/' + str(nome) + '/test.csv', header=None).values

    nameFeatures = dataTrain[0, :]
    Xtrain = retiraItens(dataTrain[1:, :-1])
    Xtest = retiraItens(dataTest[1:, :-1])
    Ytrain = dataTrain[1:, -1:]
    Ytest = dataTest[1:, -1:]

    # print("XXXX: ", Xtrain)
    # print("YYYYY: ", Ytrain)

    return nameFeatures, normalizarDados(Xtrain.astype(float)), normalizarDados(Xtest.astype(float)), Ytrain.astype(float), Ytest.astype(float)


def escolheFeatures(vetorDados, features):

    vetorFiltro = []
    print(vetorDados.T)

    for coluna in features:

        index = coluna[0]

        if coluna[1] == 1:
            vetorFiltro.append(vetorDados.T[index])

    print(np.array(vetorFiltro))
    print(np.shape(vetorFiltro))
    print(features)
    return np.array(vetorFiltro).T
