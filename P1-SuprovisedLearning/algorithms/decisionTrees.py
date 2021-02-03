#Code from: https://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


## TODO: 
## Add Cross validation 
## Add Pruning 
## Add information Gain 
## Add Graphing 

def decisionTreeClassifier(trainingDataX, TraingingDataY, testDataX, testDataY):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainingDataX, TraingingDataY)

    testdecisionTree(clf,testDataX, testDataY)


def testdecisionTree(clf, testDataX, testDataY):
    res_pred = clf.predict(testDataX)
    score = accuracy_score(testDataY.round(), res_pred)
    print(score)


def graphdecisionTree():
    print(1)

def prune():
    print(1)




