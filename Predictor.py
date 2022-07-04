import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import csv
import pandas as pd
import random

def get_tickers(csvfile):
    tickers = []

    with open(csvfile, encoding="utf-8") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        first = True
        for row in csvReader:
            if row:
                if first == False:
                    tickers.append(str(row[1]))
                else:
                    first = False
    return tickers


def clf_predictions23(X, clf, tickers,y=None, cutoff=0):

    predictions = clf.predict(X)
    pp = clf.predict_proba(X)
    #print(pp)
    if y is not None:
        score = clf.score(X, y)
        print('SCORE:::: ', score)


    ups = 0
    ups_wrong = 0
    for i in range(len(tickers)):
        #print (predictions[i])
        if pp[i][0] < cutoff:#HOLY SHIT THAT MAKES IT SO MUCH BETTER
            ups += 1

            if y is not None:
                #my own score of up down-->percentage of up that were in fact down
                if y[i] == 0:
                    ups_wrong += 1
                print (tickers[i], ' 0=DOWN, 1=UP, 2=UPMORES+P, 3=UP4*S*P: ',  predictions[i],' Real: ', y[i])
            else:
                print (tickers[i], ' 0=DOWN, 1=UP, 2=UPMORES+P, 3=UP4*S*P: ',  predictions[i])
            print (tickers[i], ' Probabilities: ',  pp[i])

    if y is not None and ups != 0:
        print ('PRECENT that at least right about moving up: ', str(1-(ups_wrong/ups)), 'NUM UP: ', ups)
    return pp

def clf_predictions_8_cats(X, clf, tickers, y=None, cutoff=0):

    predictions = clf.predict(X)
    pp = clf.predict_proba(X)
    #print(pp)
    if y is not None:
        score = clf.score(X, y)
        print('SCORE:::: ', score)


    ups = 0
    ups_wrong = 0
    for i in range(len(tickers)):
        #print (predictions[i])
        if predictions[i] > 5 and sum(pp[i][4:7]) > cutoff:
            ups += 1

            if y is not None:
                #my own score of up down-->percentage of up that were in fact down
                if y[i] < 5:
                    ups_wrong += 1
                print (tickers[i], ' Prediction: ',  predictions[i],' Real: ', y[i])
            else:
                print (tickers[i], ' Prediction: ',  predictions[i])
            print (tickers[i], ' Probabilities: ',  pp[i])

    if y is not None and ups != 0:
        print ('Prediction Accuracy: ', str(1-(ups_wrong/ups)), 'NUM UP: ', ups)
    return pp


def setup_data_categorized(csvfile, rnd, scaler=None):
                # TODO TRY RELOAD DATA FROM FILE USING BUILT IN NUMPY SHIT
    with open(csvfile, encoding="utf-8") as f:
        column_names = f.readline().split(',')
        ncols = len(f.readline().split(','))
        print(ncols)
    data = np.loadtxt(csvfile, delimiter=',', skiprows=1,
                      usecols=range(2, ncols - 1), encoding="utf-8")

    print(data.shape)
    #print(data[:10])

    # split into
    X = data[:, 1:]  # select columns 1 through end, counts
    y = data[:, 0]   # select column 0, whether the stock went up or down
    #print(y[:10])
    #print(X[:10])

    #categories out of y
    y = pd.cut(y,[-10000000,-.05,-.02,-.01,0,.01,.02,.05,10000000], labels=[1,2,3,4,5,6,7,8])
    print (f'Cateorized y: {y}')

    if scaler is None:
        #scaler = StandardScaler()
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        print ('scaled')
    else:
        X = scaler.transform(X)
        print ('scaled')

    # split the data
    from sklearn.model_selection import train_test_split #X IS X_SCALED IF NO SCALER GIVEN!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=rnd)
    return scaler, X, y, X_train, X_test, y_train, y_test 



def log_regress(X_train, X_test, y_train, y_test, C=1):

    # all parameters not specified are set to their defaults
    print('Log regress start')
    logisticRegr = LogisticRegression(max_iter=10000, C=C)#C is inverse regularization(higher)

    logisticRegr.fit(X_train, y_train)

    # Use score method to get accuracy of model
    score = logisticRegr.score(X_test, y_test)
    print('Log regress, split data, complete, score:', score)

    # then see https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
    # if want to continue with confusion matrix etc
    return logisticRegr


def neural_net(X_train, X_test, y_train, y_test, alpha=0.1):

    #NEURAL!
    print('neural net start')
    clf = MLPClassifier(hidden_layer_sizes=(20000), solver='adam', tol=1e-4, random_state=None, alpha=alpha,
                verbose=True) #default max_iter = 200
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('NEURAL NET SCORE!!!:::', score)

    return clf


def print_predicts_csv(csv_file, y, predicts, tickers):
    with open(csv_file, "w", encoding="utf-8") as text_file: # "a" for appending
        text_file.write('CategoriesTODO....')

        for i in range(len(tickers)):

            text_file.write('\n' + tickers[i]+','+str(y[i]))
            for pp in predicts:
                for p in pp[i]:
                    text_file.write(','+str(p))



if __name__ == "__main__":
    csv_file_train = 'counts_What--from2021-08-31_to_2021-10-19.csv'
    rnd = random.randint(0, 1000)
    #rnd = 0
    alpha = 1
    C = .1
    print ('RND SEED:', rnd)
    

    #initial learning on split data
    scaler, X, y, X_train, X_test, y_train, y_test = setup_data_categorized(csv_file_train, rnd)
    logReg = log_regress(X_train, X_test, y_train, y_test,C=C)#=2forplainMA-IN, 1 for S
    #nn = neural_net(X_train, X_test, y_train, y_test, alpha=alpha)#=1forplainMA-IN, 1.2 for S, cept .5 when full S data..

    #DUMP
    from joblib import dump, load
    dump(logReg, str(date.today())+'C'+str(C)+'.joblib')
    #dump(nn, str(date.today())+'alpha'+str(alpha)+'.joblib')
    dump(scaler, str(date.today())+'scaler.joblib')
    #LOAD
    #scaler = load('2021-11-02-20day4CO-NOVIX-scaler.joblib')
    #logReg = load('2021-11-02-20day4CO-NOVIX-C0.5.joblib')
    #nn = load('2021-11-02-20day-4CO-NOVIX-alpha1.joblib')

    #Predicts using split test
    print('TEST PREDICTIONS')
    tickers = get_tickers(csv_file_train)#NOT CORRECT
    #clf_predictions_8_cats(X_test, logReg, tickers, y=y_test, cutoff=0.75)

    #THEN FINAL PREDICTS
    csv_file_predict = 'counts_What--from2021-10-20_to_2021-10-26.csv'
    tickers = get_tickers(csv_file_predict)
    #REMEMBER-->THIS SCALES IT-->WHY NOT PASSED INTO PREDICTIONS!
    scaler, X, y, X_train, X_test, y_train, y_test = setup_data_categorized(csv_file_predict, rnd, scaler=scaler)
    print('LOG PREDICTIONS FINAL')
    clf_predictions_8_cats(X, logReg, tickers, y=y, cutoff=0.55)
    #print('NET PREDICTIONS FINAL')
    #clf_predictions_8_cats(X, nn, tickers, y=y, cutoff=.01)
