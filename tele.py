# -*- coding: utf-8 -*-
"""tele.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1poP-DVvuawny6vmYTKbwBjcaAVBDxCJA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

data = pd.read_csv('telecustomer.csv')
data

# --- Exploratory Data Analysis (EDA) ---
print(data.head(5))
print(data.info())
print( data.describe())
print(data.isnull().sum())
print( data['Churn'].value_counts())

# Visualizations
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.show()

sns.histplot(x='tenure', data=data, kde=True)
plt.title('Tenure Distribution')
plt.show()

sns.countplot(x='Contract', data=data)
plt.title('Contract Type Distribution')
plt.show()

sns.countplot(x='Contract', hue='Churn', data=data)
plt.title('Churn by Contract Type')
plt.show()

data['tenure_group'] = pd.cut(data['tenure'], bins=[0, 12, 24, 36, 48, 60, 72, 100], labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60-72', '72+'])
sns.countplot(x='tenure_group', hue='Churn', data=data)
plt.title('Churn by Tenure Group')
plt.show()

sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
plt.title('Churn by Monthly Charges')
plt.show()

# Create tenure groups based on ranges
bins = [0, 12, 24, 36, 48, 60, 72]  # Define tenure intervals
labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']

# Create a new column
data['tenure_group'] = pd.cut(data['tenure'], bins=bins, labels=labels, include_lowest=True)

# --- Data Preprocessing ---

# Handle Missing Values (Impute TotalCharges)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  # Convert to numeric
data['TotalCharges'] = data['TotalCharges'].fillna(data.groupby('tenure_group')['TotalCharges'].transform('median'))
data.drop(columns=['tenure_group'], inplace=True)  # Remove temporary column

# Verify no more missing values
print("\nMissing Values After Imputation:\n", data.isnull().sum())

# Identify categorical and numerical features
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove('customerID')
categorical_features.remove('Churn')
numerical_features = data.select_dtypes(include=['number']).columns.tolist()

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- Split Data ---
X = data.drop(['Churn', 'customerID'], axis=1)
y = data['Churn'].map({'Yes': 1, 'No': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Build, Train, and Evaluate Models ---

# Logistic Regression
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42, solver='liblinear'))])

# Decision Tree
pipeline_dt = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', DecisionTreeClassifier(random_state=42))])

# Random Forest
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

# Gradient Boosting (XGBoost)
from xgboost import XGBClassifier  # No need to install in Colab, it's pre-installed
pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])


# Train the models
pipeline_lr.fit(X_train, y_train)
pipeline_dt.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
pipeline_xgb.fit(X_train, y_train)

# Make predictions
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_dt = pipeline_dt.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_xgb = pipeline_xgb.predict(X_test)

# Get predicted probabilities (for models that support it)
y_proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]
y_proba_rf = pipeline_rf.predict_proba(X_test)[:, 1]
y_proba_xgb = pipeline_xgb.predict_proba(X_test)[:, 1]

# Evaluation function
def evaluate_model(y_true, y_pred, y_proba=None, model_name=""):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    if y_proba is not None:
        print("ROC AUC:", roc_auc_score(y_true, y_proba))
        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_true, y_proba):.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()


# Evaluate each model
evaluate_model(y_test, y_pred_lr, y_proba_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_dt, None, "Decision Tree")
evaluate_model(y_test, y_pred_rf, y_proba_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, y_proba_xgb, "XGBoost")

