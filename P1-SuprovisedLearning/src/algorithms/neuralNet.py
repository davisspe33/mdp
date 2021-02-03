from sklearn.neural_network import MLPClassifier
from algorithms.testScript import testAlgo


## TODO: 
## Add Cross validation 
## Add Graphing 

def neuralNet(trainingDataX, TraingingDataY, testDataX, testDataY):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(trainingDataX, TraingingDataY)
    testAlgo('ann',clf, testDataX, testDataY)