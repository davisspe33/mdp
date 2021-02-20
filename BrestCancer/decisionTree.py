
import csv
import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 

def main():
    x, y = transformCancerData()
    decisionTreeClassifier(x,y)
    #plotmodel(x,y)
    #plotmodelLeaf(x,y)

def transformCancerData(): 
    data = pd.read_csv('Cancerdata.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['diagnosis','id'], axis=1)
    le = LabelEncoder() 
    y = le.fit_transform(data['diagnosis'])
    return x,y

def decisionTreeClassifier(x,y):
    clf = tree.DecisionTreeClassifier(random_state=0, max_depth=4, max_leaf_nodes=15)
    scores = cross_val_score(clf, x, y, cv=5, verbose=1)
    print(scores)


def plotmodel(x,y):
    param_range= np.linspace(1, 10, num=10)
    train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=0), x, y, param_name="max_depth", param_range=param_range, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    lw = 2
    plt.scatter(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.scatter(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.xlabel('max_depth')
    plt.ylabel('R squared accuracy')
    plt.legend(loc="best")
    plt.show()
def plotmodelLeaf(x,y):
    param_range= np.linspace(2, 30, num=29)
    param_range= param_range.astype('int')
    train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=0, max_depth=4), x, y, param_name="max_leaf_nodes", param_range=param_range, cv=5)
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
