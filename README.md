# Implementation of Linear Regression Using Gradient Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SANJAI A
RegisterNumber: 212220040142
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m = len(y)
  h = X.dot(theta)
  square_err = (h-y)**2
  return 1/(2*m) * np.sum(square_err)

data_n = data.values
m = data_n[:,0].size
X = np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y = data_n[:,1].reshape(m,1)
theta = np.zeros((2,1))
computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m = len(y)
  J_history = []

  for i in range(num_iters):
    predictions = X.dot(theta)
    error = np.dot(X.transpose(),(predictions-y))
    descent = alpha * 1/m * error
    theta -= descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value = [x for x in range(25)]
y_value = [y*theta[1]+theta[0] for y in x_value] 
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population nof City (10,000)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(X,theta):
  predictions = np.dot(theta.transpose(),X)
  return predictions[0]

predict1 = predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2 = predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![image](https://user-images.githubusercontent.com/95969295/204101072-8bf52d15-7f6a-42a5-bb55-b540ec3d004c.png)

![image](https://user-images.githubusercontent.com/95969295/204101100-d8d0fb25-50c9-4d52-b785-00da57e5258c.png)

![image](https://user-images.githubusercontent.com/95969295/204101125-59ddf2dc-65d3-4a6f-b3ae-2499c183c946.png)

![image](https://user-images.githubusercontent.com/95969295/204101147-426f6c9b-7e40-4a5f-9625-8541df491502.png)

![image](https://user-images.githubusercontent.com/95969295/204101179-e4d8ff02-a458-4764-b89a-83b2cc94f33e.png)

![image](https://user-images.githubusercontent.com/95969295/204101206-5ced9fae-ab48-4b4f-8a3e-7431d09a0f1a.png)

![image](https://user-images.githubusercontent.com/95969295/204101225-650ee6c3-d147-40d6-9365-32c8d0ac9d0c.png)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
