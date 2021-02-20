
import csv
import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 

def main():
    x, y = transformCancerData()
    svc_rbf_kernal(x,y)
    #plotmodelC(x,y)

def transformCancerData(): 
    data = pd.read_csv('Cancerdata.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['diagnosis','id'], axis=1)
    le = LabelEncoder() 
    y = le.fit_transform(data['diagnosis'])
    return x,y

def svc_rbf_kernal(x,y):
    clf = make_pipeline(StandardScaler(), SVC(C=2))
    scores = cross_val_score(clf, x, y, cv=5, verbose=1)
    print(scores)

def plotmodelC(x,y):
    param_range= np.linspace(.1, 10, num=10)
    train_scores, test_scores = validation_curve(SVC(), x, y, param_name="C", param_range=param_range, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.xlabel('C: Regularization parameter')
    plt.ylabel('R squared accuracy')
    plt.legend(loc="best")
    plt.show()

main()
