import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def NAIVEBAYES(X_train,Y_train,X_test,Y_test):
    clf5 = GaussianNB()
    clf5=clf5.fit(X_train,np.ravel(Y_train))
    Y_pred = clf5.predict(X_test)
    print("\nNaive Bayes\n")
    print("Accuracy")
    print(metrics.accuracy_score(Y_test, Y_pred))
    print("\nConfusion matrix")
    conf_matrix=metrics.confusion_matrix(Y_test,Y_pred)
    print(conf_matrix)
    print("\nClassification Report")
    print(metrics.classification_report(Y_test,Y_pred))

    print("___________________________________________________________________________________________________________________________________")
