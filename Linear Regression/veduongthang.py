import numpy as np
import matplotlib.pyplot as plt

A = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]
b = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
plt.plot(A,b, 'ro')

#convert list A to matrix
A = np.array([A]).transpose()
# b = np.array([b]).transpose()
b = np.array(b).transpose()
ones = np.ones((A.shape[0],1), dtype = np.int8)
A = np.concatenate((A, ones), axis = 1)
x = np.linalg.inv((A.transpose().dot(A))).dot(A.transpose()).dot(b)
print(np.linalg.inv((A.transpose().dot(A))).dot(A.transpose()).shape)
print(b.shape)
print(x.shape)
x0 = np.array([[1,46]]).T
print(x0)
y0 = x[0]*x0 + x[1]

plt.plot(x0,y0)
plt.show()