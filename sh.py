import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            RocCurveDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Load data
file_path = "Autism-Prediction-using-Machine-Learning---DataSet.csv"
df = pd.read_csv(file_path)

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert categorical columns to proper format
categorical_cols = ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'relation']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

# Encode target variable
df['austim'] = df['austim'].map({'yes': 1, 'no': 0})

# Handle missing values
df.dropna(subset=['austim'], inplace=True)  # Ensure no NaN in target

# Feature Engineering
df['total_score'] = df[[f'A{i}_Score' for i in range(1, 11)]].sum(axis=1)
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 30, 50, 100],
                        labels=['child', 'teen', 'young', 'adult', 'senior'])

# Define features and target
y = df['austim']
features = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'age', 'gender', 'ethnicity', 'jaundice', 'contry_of_res',
            'used_app_before', 'result', 'relation', 'total_score', 'age_group']
X = df[features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing pipeline
numeric_features = ['age', 'result', 'total_score']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['gender', 'ethnicity', 'jaundice', 'contry_of_res',
                       'used_app_before', 'relation', 'age_group']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Feature selection
feature_selector = SelectKBest(score_func=f_classif, k=15)

# Define models
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
lgbm_model = LGBMClassifier(random_state=42)

# Voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft'
)

# Full pipeline with SMOTE
pipeline = make_imb_pipeline(
    preprocessor,
    SMOTE(random_state=42),
    feature_selector,
    voting_clf
)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save model
joblib.dump(pipeline, 'autism_detection_model.pkl')