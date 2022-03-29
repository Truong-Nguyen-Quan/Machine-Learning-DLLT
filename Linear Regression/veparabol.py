import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

A = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
b = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]

#convert A and b
A_array = np.array([A]).T
A_square = A_array*A_array
ones_array = np.ones((A_array.shape[0], 1), dtype=int)
A_array = np.concatenate((A_square, A_array, ones_array), axis=1)
b_array = np.array([b]).T

#calculate x
x = np.linalg.inv(A_array.T.dot(A_array)).dot(A_array.T).dot(b_array)

#draw parabol
x0 = np.arange(30)
y0 = x[0]*(x0*x0) + x[1]*x0 + x[2]

ax = plt.axes(xlim=(-18,46), ylim=(-18,66))
plt.plot(A,b, "ro", color="b")
plt.plot(x0,y0, color="r")

#Draw parabol by using sklearn library
A = np.concatenate((A_square, np.array([A]).T), axis=1)
b = np.array([b]).T
lr = linear_model.LinearRegression()
lr.fit(A,b)
y0_lib = lr.coef_[0][0]*(x0*x0) + lr.coef_[0][1]*x0 + lr.intercept_
plt.plot(x0,y0_lib, color="orange")

plt.show()