
import csv
import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

def main():
    x, y = transformCancerData()
    neuralNet(x,y)
    #plotmodelMaxIt(x,y)
    #plotmodelHidden(x,y)

def transformCancerData(): 
    data = pd.read_csv('Cancerdata.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['diagnosis','id'], axis=1)
    le = LabelEncoder() 
    y = le.fit_transform(data['diagnosis'])
    return x,y

def neuralNet(x,y):
    clf = MLPClassifier(max_iter=400, hidden_layer_sizes=50)
    scores = cross_val_score(clf, x, y, cv=5, verbose=1)
    print(scores)

def plotmodelHidden(x,y):
    param_range= np.linspace(1, 50, num=50)
    param_range= param_range.astype('int')
    train_scores, test_scores = validation_curve(MLPClassifier(), x, y, param_name="hidden_layer_sizes", param_range=param_range, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.xlabel('Hidden_layers')
    plt.ylabel('R squared accuracy')
    plt.legend(loc="best")
    plt.show()

def plotmodelMaxIt(x,y):
    param_range= np.linspace(1, 1000, num=50)
    param_range= param_range.astype('int')
    train_scores, test_scores = validation_curve(MLPClassifier(), x, y, param_name="max_iter", param_range=param_range, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.xlabel('max_iter')
    plt.ylabel('R squared accuracy')
    plt.legend(loc="best")
    plt.show()

main()