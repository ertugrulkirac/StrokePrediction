# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 18:33:56 2025

@author: ekirac
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer



data = pd.read_csv('healthcare-dataset-stroke-data.csv')



print(data.describe())

print(data.head(2))

print(data.tail(1))

print(data.isna().sum())

print(data['gender'].value_counts())



categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']


data['bmi'].fillna(data['bmi'].mean(), inplace=True)

X = data.drop('stroke', axis=1)
y = data['stroke']

column_transformer = ColumnTransformer(
    [('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
     ('numerical', StandardScaler(), numerical_columns)],
    remainder='passthrough')

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = column_transformer.fit_transform(X_train)
X_test = column_transformer.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}


# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print the accuracy of the model
    print(f"{model_name} Accuracy: {accuracy:.4f}")
   
   

from sklearn.model_selection import GridSearchCV

# Example for Random Forest hyperparameter tuning
rf = RandomForestClassifier()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Best parameters and model accuracy
print(f"Best Parameters for Random Forest: {grid_search.best_params_}")
print(f"Best Score for Random Forest: {grid_search.best_score_:.4f}")