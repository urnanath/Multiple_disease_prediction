import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
def RANDOMFOREST(X_train,Y_train,X_test,Y_test):

    clf3 = RandomForestClassifier(n_estimators=100)
    clf3 = clf3.fit(X_train,np.ravel(Y_train))
    Y_pred=clf3.predict(X_test)
    print("\nRandom Forest\n")
    print("Accuracy Score")
    print(metrics.accuracy_score(Y_test, Y_pred))
    print("\nConfusion matrix")
    conf_matrix=metrics.confusion_matrix(Y_test,Y_pred)
    print(conf_matrix)
    print("\nClassification Report")
    print(metrics.classification_report(Y_test,Y_pred))

    print("___________________________________________________________________________________________________________________________________")

