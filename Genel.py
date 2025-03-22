# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 14:05:33 2025

@author: ekirac
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labels = ['kedi', 'köpek', 'kuş', 'kedi', 'köpek']


le = LabelEncoder()

encoded_labels = le.fit_transform(labels)

print(encoded_labels)  # [0 1 2 0 1]


etiketler = le.inverse_transform(encoded_labels)

print(etiketler)  # ['kedi' 'köpek' 'kuş' 'kedi' 'köpek']


veri = [['kedi'], ['köpek'], ['kuş'], ['kedi']]

encoder = OneHotEncoder(sparse_output=False)

sonuc = encoder.fit_transform(veri)

print(sonuc)


from sklearn.preprocessing import StandardScaler

import numpy as np

X = np.array([[10], [20], [30], [40], [50]])

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled)
