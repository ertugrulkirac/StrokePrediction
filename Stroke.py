# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 00:33:31 2024

@author: ekirac
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

veri = pd.read_csv('healthcare-dataset-stroke-data.csv')

df = pd.DataFrame(veri)

df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

#df.iloc[0:2]  # İlk iki satırı getirir (0 ve 1. satırlar)
gender = df.iloc[:,1:2].values #cinsiyet sütunu



ohe = OneHotEncoder()

gender = ohe.fit_transform(gender).toarray()
sonuc1 = pd.DataFrame(data=gender, index=range(len(veri)), columns= ['Male','Female','Other'])


age_hip_hear = df.iloc[:,2:5].values
sonuc2 = pd.DataFrame(data=age_hip_hear, index=range(len(veri)), columns= ['age', 'hypertension', 'heart_disease'])


lbl_enc = LabelEncoder()

married = df.iloc[:,5:6].values
married = lbl_enc.fit_transform(married)


sonuc3 = pd.DataFrame(data=married, index=range(len(veri)), columns= ['ever_married'])


workType = df.iloc[:,6:7].values


workType = ohe.fit_transform(workType).toarray()

sonuc4 = pd.DataFrame(data=workType, index=range(len(veri)), columns= ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'childeren'])

residence_type = df.iloc[:,7:8].values
residence_type = lbl_enc.fit_transform(residence_type)

sonuc5= pd.DataFrame(data=residence_type, index=range(len(veri)), columns= ['residence_type'])

smoking_status = df.iloc[:,10:11].values
smoking_status = ohe.fit_transform(smoking_status).toarray()


sonuc6= pd.DataFrame(data=smoking_status, index=range(len(veri)), columns= ['Unknown', 'fomerly smoked', 'never smoked', 'smokes'])

birlestirme = pd.concat([sonuc1, sonuc2, sonuc3, sonuc4, sonuc5, sonuc6], axis=1)

stroke =  df.iloc[:,-1].values
stroke = pd.DataFrame(data=stroke, index=range(len(veri)), columns=['stroke'])

x_train, x_test, y_train, y_test = train_test_split(birlestirme, stroke, test_size=0.3, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

model =  DecisionTreeClassifier();
#model =  KNeighborsClassifier(n_neighbors=5);
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)*100
print(accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)


