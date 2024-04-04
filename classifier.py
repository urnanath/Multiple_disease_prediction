from sklearn import metrics
from sklearn import svm

def SVM(X_train,Y_train,X_test,Y_test):   
    clf = svm.SVC(kernel='linear')
    #training the support vector Machine Classifier
    clf.fit(X_train,Y_train)
    
    # accuracy score on the training data
    Y_pred = clf.predict(X_test)
    print("\nSVM model\n")
    print("Accuracy Score")
    print(metrics.accuracy_score(Y_test,Y_pred))
    print("\nConfusion matrix")
    conf_matrix=metrics.confusion_matrix(Y_test,Y_pred)
    print(conf_matrix)
    print("\nClassification Report")
    print(metrics.classification_report(Y_test,Y_pred))

    print("___________________________________________________________________________________________________________________________________")
    

    