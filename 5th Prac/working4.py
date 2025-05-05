# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------------
# Step 1: Load and Explore Data
# ----------------------------------------
df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Dataset Head:")
display(df.head())

print("\nDataset Info:")
print(df.info())

# Check class distribution
print("\nClass Distribution (Churn):")
print(df["Churn"].value_counts())

# ----------------------------------------
# Step 2: Data Preprocessing
# ----------------------------------------
# Drop irrelevant columns
df = df.drop("customerID", axis=1)

# Convert 'TotalCharges' to numeric (handle missing/blank entries)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Convert categorical variables to dummy variables (One-hot encoding)
df_encoded = pd.get_dummies(df)

# Split features (X) and target (y)
X = df_encoded.drop("Churn_Yes", axis=1)  # Use 'Churn_Yes' as target (1 if Yes, 0 if No)
y = df_encoded["Churn_Yes"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------------------
# Step 3: Train the Model
# ----------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------------------
# Step 4: Evaluate the Model
# ----------------------------------------
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
