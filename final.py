import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified path
dataset_path = 'C:/Users/KIIT/Documents/IMP DOCUMENTS/Celebal Tech/CUSTOMER CHURN/Dataset.csv'
df = pd.read_csv(dataset_path)

# Display the first few rows
print(df.head())

# Display information about the dataset
print(df.info())

# Display statistical summary of the dataset
print(df.describe())

# Drop customerID column as it's not needed for prediction
df.drop(['customerID'], axis=1, inplace=True)

# Replace 'No internet service' and 'No phone service' with 'No'
df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Define features and target variable
X = df.drop(['Churn'], axis=1)
y = df['Churn']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exploratory Data Analysis (EDA)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categoric_features = X.select_dtypes(include=['object']).columns

# Plotting distributions for numeric features
for feature in numeric_features:
    sns.histplot(df[feature], kde=True)
    plt.title(f'{feature} Distribution')
    plt.show()

# Plotting distributions for categorical features
for feature in categoric_features:
    sns.countplot(x=feature, data=df)
    plt.title(f'{feature} Distribution')
    plt.xticks(rotation=90)
    plt.show()

# Preprocessing pipelines for numeric and categorical features
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categoric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categoric_pipeline, categoric_features)
    ]
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Model definitions and evaluations
# Logistic Regression CV
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegressionCV(max_iter=1000))
])
logistic_pipeline.fit(X_train, y_train)
y_pred_logistic = logistic_pipeline.predict(X_test)
print('Logistic Regression CV:')
print('Accuracy Score:', accuracy_score(y_pred_logistic, y_test))
print(confusion_matrix(y_pred_logistic, y_test))
print(classification_report(y_pred_logistic, y_test))

# AdaBoost Classifier
adaboost_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier())
])
adaboost_pipeline.fit(X_train, y_train)
y_pred_adaboost = adaboost_pipeline.predict(X_test)
print('AdaBoost Classifier:')
print('Accuracy Score:', accuracy_score(y_pred_adaboost, y_test))
print(confusion_matrix(y_pred_adaboost, y_test))
print(classification_report(y_pred_adaboost, y_test))

# Random Forest Classifier
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
random_forest_pipeline.fit(X_train, y_train)
y_pred_random_forest = random_forest_pipeline.predict(X_test)
print('Random Forest Classifier:')
print('Accuracy Score:', accuracy_score(y_pred_random_forest, y_test))
print(confusion_matrix(y_pred_random_forest, y_test))
print(classification_report(y_pred_random_forest, y_test))

# Support Vector Machine
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', random_state=42))
])
svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)
print('Support Vector Machine:')
print('Accuracy Score:', accuracy_score(y_pred_svm, y_test))
print(confusion_matrix(y_pred_svm, y_test))
print(classification_report(y_pred_svm, y_test))
