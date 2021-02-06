#Code from: https://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
from algorithms.testScript import testAlgo
from sklearn.model_selection import cross_val_score


## TODO: 
## Add Cross validation 
## Add Pruning 
## Add information Gain 
## Add Graphing 

def decisionTreeClassifier(trainingDataX, TraingingDataY, testDataX, testDataY):
    clf = tree.DecisionTreeClassifier()
    return clf
    clf = clf.fit(trainingDataX, TraingingDataY)
    testAlgo('dt', clf, testDataX, testDataY)




