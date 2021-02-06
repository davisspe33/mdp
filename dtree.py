
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

def main():
    x, y = transformCreditData()
    decisionTreeClassifier(x,y)

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
    y=y/100
    y=y.round()
    y=y*100
    return x,y


def decisionTreeClassifier(x,y):
    clf = tree.DecisionTreeClassifier(random_state=10, max_depth=3)
    scores = cross_val_score(clf, x, y, cv=5)
    #scoring = {'prec_macro': 'precision_macro', 'rec_macro': make_scorer(recall_score, average='macro')}
    #scores = cross_validate(clf, x, y, scoring=scoring, cv=5, return_train_score=True)

    print(scores)
main()
