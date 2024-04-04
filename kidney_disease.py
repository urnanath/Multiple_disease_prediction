import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from classifier import SVM
from classifier2 import KNN
from classifier3 import RANDOMFOREST
from classifier4 import DECISIONTREE
from classifier5 import NAIVEBAYES

def  KIDNEY_DISEASE():
    kidney_dataset=pd.read_csv('D:\\c course\\disease-prediction-model\\dataset\\kidney_disease.csv')
    print(kidney_dataset.head())

    # removing unnecessary columns 
    kidney_dataset.pop('id')


    print(kidney_dataset.info())
    # converting necessary columns to numerical type which are in object type  

    kidney_dataset['packed_cell_volume'] = pd.to_numeric(kidney_dataset['packed_cell_volume'], errors='coerce')
    kidney_dataset['white_blood_cell_count'] = pd.to_numeric(kidney_dataset['white_blood_cell_count'], errors='coerce')
    kidney_dataset['red_blood_cell_count'] = pd.to_numeric(kidney_dataset['red_blood_cell_count'], errors='coerce')
    print(kidney_dataset.info())


    # Extracting categorical and numerical columns

    cat_cols = [col for col in kidney_dataset.columns if kidney_dataset[col].dtype == 'object']
    num_cols = [col for col in kidney_dataset.columns if kidney_dataset[col].dtype != 'object']

    # filling null values, we will use two methods, random sampling for higher null values and 
    # mean/mode sampling for lower null values

    def random_value_imputation(feature):
        random_sample = kidney_dataset[feature].dropna().sample(kidney_dataset[feature].isna().sum())
        random_sample.index = kidney_dataset[kidney_dataset[feature].isnull()].index
        kidney_dataset.loc[kidney_dataset[feature].isnull(), feature] = random_sample
    
    def impute_mode(feature):
        mode = kidney_dataset[feature].mode()[0]
        kidney_dataset[feature] = kidney_dataset[feature].fillna(mode)

    # filling num_cols null values using random sampling method

    for col in num_cols:
        random_value_imputation(col)

    # filling "red_blood_cells" and "pus_cell" using random sampling method and rest of cat_cols using mode imputation

    random_value_imputation('red_blood_cells')
    random_value_imputation('pus_cell')

    for col in cat_cols:
        impute_mode(col)

    #category encoding 
    le = LabelEncoder()
    for col in cat_cols:
        kidney_dataset[col] = le.fit_transform(kidney_dataset[col])    

    print(kidney_dataset.head())   
    
    #separating data and labels
    X = kidney_dataset.drop(columns = 'classification', axis=1)
    Y = kidney_dataset['classification']
    # print(X)
    # print(Y)

     #standardizing the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # print(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    # print(X.shape, X_train.shape, X_test.shape)

    print("KIDNEY DISEASE")
    SVM(X_train,Y_train,X_test,Y_test)
    KNN(X_train,Y_train,X_test,Y_test)
    RANDOMFOREST(X_train,Y_train,X_test,Y_test)
    DECISIONTREE(X_train,Y_train,X_test,Y_test)
    NAIVEBAYES(X_train,Y_train,X_test,Y_test)
    print("----------------------------------------------------------------------------------------------------------------------------------------------")