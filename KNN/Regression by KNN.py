import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def linearline_library(A,b,x0):
	lr = linear_model.LinearRegression()
	lr.fit(A,b)
	y0 = lr.coef_*x0 + lr.intercept_
	return y0

def knn_library(A,b,x0,k):
	knn = neighbors.KNeighborsClassifier(n_neighbors = k)
	knn.fit(A,b)
	y0 = knn.predict(x0)
	return y0

#Return a value
def cost(A,b,x):
	m = A.shape[0]
	return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2

#Return a matrix
def grad(A,b,x):
	m = A.shape[0]
	return 1/m * A.T.dot(A.dot(x) - b)

def check_grad(A,b,x):
	eps = 1e-4
	g_eps = np.zeros_like(x)
	for i in range(len(x)):
		x1 = x.copy()
		x2 = x.copy()
		x1[i] += eps
		x2[i] -= eps
		g_eps[i] = (cost(A,b,x1) - cost(A,b,x2))/(2 * eps)
	g_formula = grad(A,b,x)
	if np.linalg.norm(g_formula - g_eps) > 1e-5:
		print("WARNING ABOUT GRADIENT FUNCTION!")

def gradient_descent(A, b, x_init, learning_rate, iteration):
	x_list = [x_init]
	for i in range(iteration):
		x_new = x_list[-1] - learning_rate*grad(A,b,x_list[-1])
		#When to end the GD loop
		if np.linalg.norm(grad(A,b,x_new), 2) * 1/len(x_new) < 1:
			break
		x_list.append(x_new)
	return x_list

def main():
	A = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,36,40,46]]).T
	b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T
	plt.plot(A,b,'ro')
	plt.title("Regression by Linear Regression, KNN and GD")

	x0 = np.array([[1,46]]).T

	#calculate y0 by linear regression
	y0 = linearline_library(A,b,x0)
	plt.plot(x0,y0,color = 'g')
	print(y0)

	#calculate y0 by KNN
	y0 = knn_library(A,b.reshape(-1),x0,3)
	plt.plot(x0,y0,color = 'b')
	print(y0)

	#calculate y0 by GD
	ones = np.ones((A.shape[0], 1), dtype=int)
	A_GD = np.concatenate((A, ones), axis=1)

	x_init = np.array([[1.0],[2.0]])
	check_grad(A_GD,b,x_init)
	y0_init = x_init[0][0]*A + x_init[1][0]
	plt.plot(A,y0_init, color="black")

	iteration = 100
	learning_rate = 0.0001

	x_list = gradient_descent(A_GD, b, x_init, learning_rate, iteration)

	for i in range(len(x_list)):
		y0_x_list = x_list[i][0] * A + x_list[i][1]
		plt.plot(A, y0_x_list, color = "black", alpha=0.1)
	y0 = x_list[-1][0] * x0 + x_list[-1][1]
	print(y0)

	plt.legend(["points", "linear", "KNN", "GD"], loc="upper left")
	plt.show()

main()