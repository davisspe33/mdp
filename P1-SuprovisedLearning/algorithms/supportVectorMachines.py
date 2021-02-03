from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


## TODO: 
## Add Cross validation 
## Add Graphing 

def svm(trainingDataX, TraingingDataY, testDataX, testDataY):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(trainingDataX, TraingingDataY)
    testsvm(clf, testDataX, testDataY)
    
def testsvm(clf, testDataX, testDataY):
    res_pred = clf.predict(testDataX)
    score = accuracy_score(testDataY.round(), res_pred)
    print(score)
