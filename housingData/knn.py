
import csv
import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

def main():
    x, y = transformCreditData()
    knn(x,y)
    plotmodel(x,y)

def transformCreditData(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    y = data['MEDV']

    return x,y

def knn(x,y):
    clf = KNeighborsRegressor(n_neighbors=10)
    scores = cross_val_score(clf, x, y, cv=5, verbose=1) #score is uniform average
    print(scores)

def plotmodel(x,y):
    param_range= np.linspace(1, 20, num=20)
    param_range= param_range.astype('int')
    train_scores, test_scores = validation_curve(KNeighborsRegressor(), x, y, param_name="n_neighbors", param_range=param_range, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.xlabel('Number of neighbors')
    plt.ylabel('R squared accuracy')
    plt.legend(loc="best")
    plt.show()

main()
