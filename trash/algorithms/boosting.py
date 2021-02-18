from sklearn.ensemble import GradientBoostingClassifier
from algorithms.testScript import testAlgo

## TODO: 
## Add Cross validation 
## Add Graphing 

def boosting(trainingDataX, TraingingDataY, testDataX, testDataY):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(trainingDataX, TraingingDataY)
    testAlgo('boosting',clf, testDataX, testDataY)
    

