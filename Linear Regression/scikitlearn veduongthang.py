import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

A = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,36,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T

plt.plot(A,b, 'ro')

lr = linear_model.LinearRegression()
lr.fit(A,b)

x0 = np.array([[1,46]]).T
y0 = lr.coef_*x0 + lr.intercept_
print(x0)
print(y0)

plt.plot(x0,y0)

plt.show()