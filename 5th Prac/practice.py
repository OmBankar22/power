# Practical Number 04 ---> Data Classification Using Logistic Regression

# 1. Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 2. Load data 

df = pd.read_csv(" ")

df.head()
df.info()
df["Churn"].value_counts()

# 3. Data Preprocessing 

df = df.drop("customerID",axis=1)

df["TargetCharge"] = pd.to_numeric(df["TargetCharges"],errors='coerce')

df = df.dropna()

df_encode = pd.get_dummies(df)

x = df_encode.drop("Churn_Yes",axis=1)
y = df_encode["Churn_Yes"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# 3. Train the model

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

#  4. Evaluate the model 
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("accuracy: {accuracy:.2f}%")

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test,y_pred))

