
import csv
import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

def main():
    x, y = transformCreditData()
    decisionTreeRegressor(x,y)
    plotmodel(x,y)

def transformCreditData(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    y = data['MEDV']
    return x,y

def decisionTreeRegressor(x,y):
    clf = tree.DecisionTreeRegressor(random_state=0, max_depth=3)
    #clf = tree.DecisionTreeRegressor()
    scores = cross_val_score(clf, x, y, cv=5, verbose=1)
    print(scores)

def plotmodel(x,y):
    param_range= np.linspace(1, 20, num=20)
    train_scores, test_scores = validation_curve(tree.DecisionTreeRegressor(random_state=0), x, y, param_name="max_depth", param_range=param_range, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    lw = 2
    plt.scatter(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.scatter(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.xlabel('max_depth')
    plt.ylabel('R squared accuracy')
    plt.legend(loc="best")
    plt.show()
main()
