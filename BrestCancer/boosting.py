import csv
import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import learning_curve

def main():
    x, y = transformCancerData()
    boosting(x,y)
    #plotmodel(x,y)
    #plotmodelLearn(x,y)
    

def transformCancerData(): 
    data = pd.read_csv('Cancerdata.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['diagnosis','id'], axis=1)
    le = LabelEncoder() 
    y = le.fit_transform(data['diagnosis'])
    return x,y

def boosting(x,y):
    clf = GradientBoostingClassifier(max_leaf_nodes=15)
    scores = cross_val_score(clf, x, y, cv=5, verbose=1)
    print(scores)
    learning_curve(clf,x,y)

def plotmodel(x,y):
    param_range= np.linspace(2, 30, num=29)
    param_range= param_range.astype('int')
    train_scores, test_scores = validation_curve(GradientBoostingClassifier(), x, y, param_name="max_leaf_nodes", param_range=param_range, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.xlabel('number of neighbors')
    plt.ylabel('R squared accuracy')
    plt.legend(loc="best")
    plt.show()

def plotmodelLearn(x,y):
    train_scores, test_scores = learning_curve(GradientBoostingClassifier(), x, y)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    lw = 2
    plt.plot(train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.plot(test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.xlabel('number of neighbors')
    plt.ylabel('R squared accuracy')
    plt.legend(loc="best")
    plt.show()

main()
