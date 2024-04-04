import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from classifier import SVM
from classifier2 import KNN
from classifier3 import RANDOMFOREST
from classifier4 import DECISIONTREE
from classifier5 import NAIVEBAYES

def BREAST_CANCER():
    breast_cancer_dataset=pd.read_csv('D:\\c course\\disease-prediction-model\\dataset\\breast_cancer.csv')

    print(breast_cancer_dataset.head())

    breast_cancer_dataset.pop('id')
    breast_cancer_dataset.pop('Unnamed: 32')

    print(breast_cancer_dataset.head())

    breast_cancer_dataset['diagnosis'] = breast_cancer_dataset['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)
    # print(breast_cancer_dataset.head())


    # #separating data and labels
    X = breast_cancer_dataset.drop(columns = 'diagnosis', axis=1)
    Y = breast_cancer_dataset['diagnosis']

    # print(X)
    # print(Y)

    # #standardizing the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # # print(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    # # print(X.shape, X_train.shape, X_test.shape)

    print("BREAST CANCER")
    SVM(X_train,Y_train,X_test,Y_test)
    KNN(X_train,Y_train,X_test,Y_test)
    RANDOMFOREST(X_train,Y_train,X_test,Y_test)
    DECISIONTREE(X_train,Y_train,X_test,Y_test)
    NAIVEBAYES(X_train,Y_train,X_test,Y_test)
    print("----------------------------------------------------------------------------------------------------------------------------------------------")