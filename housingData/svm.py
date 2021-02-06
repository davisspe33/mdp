
import csv
import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def main():
    x, y = transformCreditData()
    svm(x,y)

def transformCreditData(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    y = data['MEDV']
    return x,y

def svm(x,y):
    clf = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    scores = cross_val_score(clf, x, y, cv=5)
    print(scores)
main()
