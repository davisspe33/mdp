
import csv
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor

def main():
    x, y = transformCreditData()
    neuralNet(x,y)

def transformCreditData(): 
    data = pd.read_csv('HousingData.csv') 
    data = data.fillna(0)
    x = data
    x = x.drop(['MEDV'], axis=1)
    y = data['MEDV']
    y = y.astype('int')
    return x,y

def neuralNet(x,y):
    clf = MLPRegressor(random_state=1, max_iter=500)
    scores = cross_val_score(clf, x, y, cv=5)
    print(scores)
main()