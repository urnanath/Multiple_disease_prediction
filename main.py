from diabetes import DIABETES
from heart import HEART
from kidney_disease import KIDNEY_DISEASE
from breast_cancer import BREAST_CANCER

DIABETES()
HEART()
KIDNEY_DISEASE()
BREAST_CANCER()
























# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# from classifier import SVM
# from classifier2 import KNN
# from classifier3 import randomforest
# from classifier4 import decisiontree
# from classifier5 import NaiveBayes


# diabetes_dataset=pd.read_csv('D:\\c course\\disease-prediction-model\\dataset\\diabetes.csv')
# print(diabetes_dataset.head())

# #separating data and labels
# X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
# Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)

# #standardizing the data
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# print(X)


# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)


# SVM(X_train,Y_train,X_test,Y_test)
# KNN(X_train,Y_train,X_test,Y_test)
# randomforest(X_train,Y_train,X_test,Y_test)
# decisiontree(X_train,Y_train,X_test,Y_test)
# NaiveBayes(X_train,Y_train,X_test,Y_test)



