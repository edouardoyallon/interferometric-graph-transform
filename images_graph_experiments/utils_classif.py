
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

def SVM(labels_train, features_train, labels_test, features_test,C):
    features_train = features_train.reshape(features_train.shape[0],-1)
    features_test = features_test.reshape(features_test.shape[0], -1)

    classifier = LinearSVC(C=C)
    scaler = StandardScaler()
    scaler.fit(features_train)  # only compute mean and std here
    features_test = scaler.transform(features_test)
    features_train = scaler.transform(features_train)

    classifier.fit(features_train, labels_train)

    yHatTrain = classifier.predict(features_train)

    accTrain = np.sum(labels_train == yHatTrain) / features_train.shape[0]


    print('C: '+str(C))
    print('Train acc: ' + str(accTrain))

    yHatTest = classifier.predict(features_test)

    accValid = np.sum(labels_test == yHatTest) / features_test.shape[0]
    print('Test acc: '+str(accValid))

    return accValid