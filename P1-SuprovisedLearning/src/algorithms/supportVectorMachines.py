from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from algorithms.testScript import testAlgo


## TODO: 
## Add Cross validation 
## Add Graphing 

def svm(trainingDataX, TraingingDataY, testDataX, testDataY):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(trainingDataX, TraingingDataY)
    testAlgo('svm',clf, testDataX, testDataY)
