import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score

def KNN(X_train,Y_train,X_test,Y_test):
    clf2=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    clf2=clf2.fit(X_train,np.ravel(Y_train))
    Y_pred=clf2.predict(X_test)
    print("\nkNearest Neighbour\n")
    print("Accuracy Score")
    print(accuracy_score(Y_test, Y_pred))
    print("\nConfusion matrix")
    conf_matrix=confusion_matrix(Y_test,Y_pred)
    print(conf_matrix)
    print("\nClassification Report")
    print(metrics.classification_report(Y_test,Y_pred))

    print("___________________________________________________________________________________________________________________________________")
