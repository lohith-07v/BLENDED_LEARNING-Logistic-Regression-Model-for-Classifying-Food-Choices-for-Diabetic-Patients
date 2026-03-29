# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load the dataset, separate input features and target class, then normalize features and encode target labels.
2. Split the dataset into training and testing sets using stratified sampling.
3. Train a Logistic Regression model with L2 regularization on the training data and predict the test data.
4. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: Lohith V
RegisterNumber:  25013313


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('food_items (1).csv')
print('Name: Lohith V')
print('Reg.No: 25013313')
print('Dataset Overview')
print(df.head())
print("\nDataset Info:")
print(df.info())
X_raw=df.iloc[:,:-1]
y_raw=df.iloc[:,-1:]
scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)
penalty='l2'
multi_class='multinomial'
solver='lbfgs'
max_iter=1000
l2_model=LogisticRegression(random_state=123,penalty=penalty,multi_class=multi_class,solver=solver,max_iter=max_iter)
l2_model.fit(X_train,y_train)
y_pred=l2_model.predict(X_test)
print('Name: Lohith V')
print('Reg. No: 25013313')
print("\nModel Evaluation:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
print("Name: Lohith V")
print("Reg. No: 25013313")
*/
*/
```

## Output:
<img width="785" height="509" alt="Screenshot 2026-03-29 144459" src="https://github.com/user-attachments/assets/de168cf5-ee24-4c1b-985f-0a8771a9943a" />

<img width="552" height="464" alt="Screenshot 2026-03-29 144508" src="https://github.com/user-attachments/assets/956e586b-ad41-423b-b169-13245405d5b2" />

<img width="626" height="382" alt="Screenshot 2026-03-29 144516" src="https://github.com/user-attachments/assets/0f972c38-dcd5-4873-9fee-f22adb713abe" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
