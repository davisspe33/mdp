from sklearn.neighbors import KNeighborsClassifier
from algorithms.testScript import testAlgo

## TODO: 
## Add Cross validation 
## Add Graphing 

def knn(trainingDataX, TraingingDataY, testDataX, testDataY):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(trainingDataX, TraingingDataY)
    testAlgo('knn',clf, testDataX, testDataY)
