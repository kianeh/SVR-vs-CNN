# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as imbpipeline

# 1) Load Dataset
df = pd.read_excel(r"C:\Users\kiane\Downloads\Bank_Personal_Loan_Modelling.xlsx", sheet_name='Data')

print(df.columns)
# 2) Data Selection and Checking
print("Dataset Shape:", df.shape)  # Prints Dataset Shape
print(df.info())
df.describe(include="all")  # Prints statistical summary for all columns (including object types)

# Check for missing and duplicated data
print("Total Number of Missing Values in Dataset is:", df.isna().sum().sum())
print("Total Number of Duplicated Rows in Dataset is:", df.duplicated().sum())

# 3) Exploratory Data Analysis (EDA)
# List of columns you want to select
columns_to_select = ['ID', 'Age', 'Experience', 'Income', 'Zip code', 'Family', 'CCAvg',
                     'Education', 'Mortgage', 'Securities Account', 'CD Account', 
                     'Online', 'CreditCard', 'Personal Loan']

# Verify if all columns exist in the DataFrame before selecting them
missing_columns = [col for col in columns_to_select if col not in df.columns]
if len(missing_columns) == 0:
    df = df[columns_to_select]  # Select only the columns that exist
else:
    print(f"These columns are missing from the DataFrame: {missing_columns}")

# 4) Drop Columns (Check if 'ID' and 'Zipcode' exist before dropping them)
columns_to_drop = ['ID', 'Zip code']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)  # Drop column if it exists
    else:
        print(f"'{col}' not found in the dataset and will not be dropped.")

# 5) Correlation Heatmap (EDA)
numeric_df = df.select_dtypes(include='number')  # Select only numeric columns

fig, ax = plt.subplots(figsize=(10, 5))
plt.rcParams.update({'font.size': 10})
fields_correlation = sns.heatmap(numeric_df.corr(), vmin=-1, cmap="PuBu", annot=True, ax=ax)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 6) Train-Test Split
X = df.drop('Personal Loan', axis=1)  # Features (excluding the target variable)
y = df['Personal Loan']  # Target variable (binary classification)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 7) Build the Pipeline (Preprocessing + Model)
# StandardScaler for feature scaling, SVC for classification
pipeline = imbpipeline(steps=[
    ('scaler', StandardScaler()),  # Standardizes the data
    ('svc', SVC())  # Support Vector Classifier model
])

# 8) Hyperparameter Tuning using GridSearchCV
# Define parameter grid for SVC
param_grid = {
    'svc__C': [0.1, 1, 10, 100],  # Regularization parameter
    'svc__kernel': ['linear', 'rbf'],  # Kernel types
    'svc__gamma': ['scale', 'auto']  # Gamma parameter for non-linear kernels
}

# Perform grid search using 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# 9) Best Parameters
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# 10) Model Evaluation on Training and Testing Set
y_train_pred = grid_search.predict(X_train)  # Predictions on training set
y_test_pred = grid_search.predict(X_test)  # Predictions on testing set

# 11) Model Performance Metrics
# Accuracy, precision, recall, f1 score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)

train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Train Precision: {train_precision}")
print(f"Test Precision: {test_precision}")
print(f"Train Recall: {train_recall}")
print(f"Test Recall: {test_recall}")
print(f"Train F1 Score: {train_f1}")
print(f"Test F1 Score: {test_f1}")

# 12) Confusion Matrix and Classification Report for Test Set
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Optional: Visualizing the Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
