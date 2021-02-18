
import csv
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


def main():
    x, y = transformCreditData()
    neuralNet(x,y)

def transformCreditData(): 
    le = LabelEncoder() 
    data = pd.read_csv('creditCardLimits.csv') 
    data['Attrition_Flag']= le.fit_transform(data['Attrition_Flag']) 
    data['Gender']= le.fit_transform(data['Gender']) 
    data['Education_Level']= le.fit_transform(data['Education_Level']) 
    data['Marital_Status']= le.fit_transform(data['Marital_Status']) 
    data['Income_Category']= le.fit_transform(data['Income_Category']) 
    data['Card_Category']= le.fit_transform(data['Card_Category']) 

    columnTransformer = ColumnTransformer([('encoder', 
                                        OneHotEncoder(), 
                                        [0,2,4,5,6])], 
                                      remainder='passthrough')
  
    x = data.filter(['Attrition_Flag','Customer_Age','Gender','Dependent_count','Education_Level','Marital_Status','Card_Category'], axis=1)        
    x= np.array(columnTransformer.fit_transform(x), dtype = np.str) 
    y = data['Credit_Limit']
    y=y/10
    y=y.round()
    y=y*10
    return x,y


def neuralNet(x,y):
    clf = MLPRegressor(random_state=1, max_iter=500)
    scores = cross_val_score(clf, x, y, cv=5)
    print(scores)

main()