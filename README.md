# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values. 
3. Import linear regression from sklearn. 
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:KOPPALA NAVEEN 
RegisterNumber: 212223100023
*/
```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y = Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)

```

## Output:

## DATA SET

![image](https://github.com/user-attachments/assets/830272ba-ab33-4d67-8046-5b8311dd8e51)

## DATA TYPES

![image](https://github.com/user-attachments/assets/dcd5e8f7-5b26-4c80-8c82-f8958a5bef05)

## NEW DATA SET

![image](https://github.com/user-attachments/assets/275d2362-84f9-42c2-99c1-12e61f474226)

## Y VALUES

![image](https://github.com/user-attachments/assets/4d7348c7-33b1-4518-9da6-e7fee501983d)

## ACCURACY

![image](https://github.com/user-attachments/assets/ce34990c-24d0-4d90-8b4c-613f74e4eed1)


## Y PRED

![image](https://github.com/user-attachments/assets/70c048a1-8aa8-4f29-baf6-8b1d09db2b1f)

## NEW Y

![image](https://github.com/user-attachments/assets/23e05df9-d4f7-4f15-8cfc-29a0e9045467)



![image](https://github.com/user-attachments/assets/698121ec-a5b1-418a-bd23-a6fa6fb29f90)


![image](https://github.com/user-attachments/assets/c3f10015-3185-47aa-a9be-35ad6848a24d)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

