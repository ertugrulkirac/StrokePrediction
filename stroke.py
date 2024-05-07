# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:43:53 2024

@author: ertugrulkirac
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.svm import SVC

data = pd.read_csv('healthcare-dataset-stroke-data.csv')

print(data.head())

categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

data['bmi'].fillna(data['bmi'].mean(), inplace=True)

X = data.drop('stroke', axis=1)
y = data['stroke']

column_transformer = ColumnTransformer(
    [('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
     ('numerical', StandardScaler(), numerical_columns)],
    remainder='passthrough')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = column_transformer.fit_transform(X_train)
X_test = column_transformer.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)*100

print(f"Accuracy Score: {accuracy:.2f}")
print(confusion_matrix(y_test, y_pred))
