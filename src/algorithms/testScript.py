from sklearn.metrics import accuracy_score

def testAlgo(algo, clf, testDataX, testDataY):
    res_pred = clf.predict(testDataX)
    score = accuracy_score(testDataY.round(), res_pred)
    print(algo)
    print(score*100)
