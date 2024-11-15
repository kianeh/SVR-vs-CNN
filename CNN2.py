# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load Dataset and basic checks
df = pd.read_excel(r"C:\Users\kiane\Downloads\Bank_Personal_Loan_Modelling.xlsx", sheet_name='Data')

# Basic insights of the Dataset
print("Dataset Shape:", df.shape)
df.describe(include="all")

# Check missing values and duplicates
print("Total Number of Missing Values in Dataset:", df.isna().sum().sum())
print("Total Number of Duplicated Rows in Dataset:", df.duplicated().sum())

# Check column names
print(df.columns)

# 2) Preprocessing the Data

# Map categorical values in 'Family' and 'Education' to integers
df['Family'] = df['Family'].astype(int)
df['Education'] = df['Education'].astype(int)

# Define features (X) and target variable (y)
X = df.drop(["Personal Loan", "ZIP Code"], axis=1)  # Remove 'Personal Loan' (target) and 'Zip code'
y = df["Personal Loan"]  # Target column

# 3) Handling Class Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, stratify=y_res, random_state=42)

# One-hot encode the target variable
y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

# Reshape the input for Conv1D (1D Convolution requires a 3D input: [samples, time_steps, features])
x_train = np.expand_dims(x_train, axis=2)  # Add a new axis
x_test = np.expand_dims(x_test, axis=2)

# 4) CNN Model Design
model = keras.models.Sequential([
    keras.layers.Conv1D(32, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)),  # Conv layer with 32 filters
    keras.layers.MaxPooling1D(pool_size=2),  # Max Pooling layer
    keras.layers.Conv1D(64, kernel_size=2, activation='relu'),  # Another Conv layer
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),  # Flatten the output for Dense layers
    keras.layers.Dense(64, activation='relu'),  # Fully connected layer
    keras.layers.Dense(2, activation='softmax')  # Output layer for binary classification (2 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use 'categorical_crossentropy' for multi-class classification
              metrics=['accuracy'])

# 5) Model Training
history = model.fit(
    x_train, y_train_oh, 
    validation_data=(x_test, y_test_oh), 
    epochs=100, 
    batch_size=8,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# 6) Model Evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test_oh)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 7) Additional Metrics Calculation

# Predictions
y_pred_oh = model.predict(x_test)
y_pred = np.argmax(y_pred_oh, axis=1)

# Convert one-hot back to categorical for true labels
y_test_cat = np.argmax(y_test_oh, axis=1)

# Precision, Recall, F1 Score
precision = precision_score(y_test_cat, y_pred)
recall = recall_score(y_test_cat, y_pred)
f1 = f1_score(y_test_cat, y_pred)

# AUC Score
roc_auc = roc_auc_score(y_test_cat, y_pred)

print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"ROC AUC Score: {roc_auc * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_cat, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_cat, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 8) Plotting Accuracy and Loss Curves

# Plot training & validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()






