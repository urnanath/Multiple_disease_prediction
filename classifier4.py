import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def DECISIONTREE(X_train,Y_train,X_test,Y_test):
    clf4 = DecisionTreeClassifier()
    clf4 = clf4.fit(X_train,np.ravel(Y_train))
    Y_pred=clf4.predict(X_test)
    print("\nDecision Tree\n")
    print("Accuracy Score")
    print(metrics.accuracy_score(Y_test, Y_pred))
    print("\nConfusion matrix")
    conf_matrix=metrics.confusion_matrix(Y_test,Y_pred)
    print(conf_matrix)
    print("\nClassification Report")
    print(metrics.classification_report(Y_test,Y_pred))

    print("___________________________________________________________________________________________________________________________________")

