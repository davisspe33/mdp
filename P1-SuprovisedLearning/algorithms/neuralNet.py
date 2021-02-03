from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


## TODO: 
## Add Cross validation 
## Add Graphing 

def neuralNet(trainingDataX, TraingingDataY, testDataX, testDataY):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(trainingDataX, TraingingDataY)
    testneuralNet(clf, testDataX, testDataY)

def testneuralNet(clf, testDataX, testDataY):
    res_pred = clf.predict(testDataX)
    score = accuracy_score(testDataY.round(), res_pred)
    print(score)