
import csv
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

from algorithms.decisionTrees import decisionTreeClassifier 
from algorithms.neuralNet import neuralNet 
from algorithms.supportVectorMachines import svm 


def main():
    x, y = transformCreditData()
    trainingDataX =  x[::int(len(x)*.75)]
    trainingDataY =  y[::int(len(y)*.75)]
    testingDataX =  x[int(len(x)*.75)::]
    testingDataY =  y[int(len(y)*.75)::]


    decisionTreeClassifier(trainingDataX,trainingDataY, testingDataX, testingDataY)
    neuralNet(trainingDataX,trainingDataY, testingDataX, testingDataY)
    svm(trainingDataX,trainingDataY, testingDataX, testingDataY)

    # algos=['']
    # for algo in algos:
    #     trainAlgo(algo, trainingData)
    #     testAlgo(algo, testingData)

def transformCreditData(): 
    le = LabelEncoder() 
    data = pd.read_csv('creditCardLimits.csv') 
    data['Attrition_Flag']= le.fit_transform(data['Attrition_Flag']) 
    data['Gender']= le.fit_transform(data['Gender']) 
    data['Education_Level']= le.fit_transform(data['Education_Level']) 
    data['Marital_Status']= le.fit_transform(data['Marital_Status']) 
    data['Income_Category']= le.fit_transform(data['Income_Category']) 
    data['Card_Category']= le.fit_transform(data['Card_Category']) 

    #commenting out One-hot-encoder
    # columnTransformer = ColumnTransformer([('encoder', 
    #                                     OneHotEncoder(), 
    #                                     [0,2,4,5,6])], 
    #                                   remainder='passthrough')
  
    x = data.filter(['Attrition_Flag','Customer_Age','Gender','Dependent_count','Education_Level','Marital_Status','Card_Category'], axis=1)        
    #x= np.array(columnTransformer.fit_transform(x), dtype = np.str) 
    y = data['Credit_Limit']
    y=y/10
    y=y.round()
    y=y*10
   
    return x,y

#credit data 
# ([('CLIENTNUM', '768805383'),
#  ('Attrition_Flag', 'Existing Customer'), 
# ('Customer_Age', '45'),
#  ('Gender', 'M'), 
# ('Dependent_count', '3'), 
# ('Education_Level', 'High School'), 
# ('Marital_Status', 'Married'), 
# ('Income_Category', '$60K - $80K'), 
# ('Card_Category', 'Blue'), 
# ('Months_on_book', '39'),
#  ('Total_Relationship_Count', '5'),
#  ('Months_Inactive_12_mon', '1'), 
# ('Contacts_Count_12_mon', '3'), 
# ('Credit_Limit', '12691'),
#  ('Total_Revolving_Bal', '777'),
#  ('Avg_Open_To_Buy', '11914'), 
# ('Total_Amt_Chng_Q4_Q1', '1.335'), 
# ('Total_Trans_Amt', '1144'), ('Total_Trans_Ct', '42'),
#  ('Total_Ct_Chng_Q4_Q1', '1.625'), ('Avg_Utilization_Ratio', '0.061'), ('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', '9.3448e-05'), ('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2', '0.99991')])
main()
