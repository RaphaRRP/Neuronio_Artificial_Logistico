import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
import math

#############################

np.random.seed(42)
ages = np.random.randint(low=15, high=70, size=40)

#############################

labels = []
for age in ages:
    if age < 30:
        labels.append(0)
    else:
        labels.append(1)

#############################

for i in range(0, 3):
    r = np.random.randint(0, len(labels) -1)
    if labels[r] == 0:
        labels[r] = 1
    else:
        labels[r] = 0
        
#############################


model = LogisticRegression()
model.fit(ages.reshape(-1, 1), labels)
m = model.coef_[0][0]
b = model.intercept_[0]
def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a
 ################
     
x = np.arange(0, 70, 0.1)
sig = sigmoid(m * x + b)
limiar_idade = abs(b / m)
#plt.plot(ages, ages * m + b, color = "blue")    
plt.plot([limiar_idade, limiar_idade], [0, 0.5], '--', color = "green")    
plt.scatter(ages, labels, color="red")
plt.plot(x, sig)
plt.show()

#############################