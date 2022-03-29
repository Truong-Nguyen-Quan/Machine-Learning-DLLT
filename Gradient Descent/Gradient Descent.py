import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.animation as animation

#Return a value
def cost(x):
	m = A.shape[0]
	return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2

#Return a matrix
def grad(x):
	m = A.shape[0]
	return 1/m * A.T.dot(A.dot(x) - b)

def check_grad(x):
	eps = 1e-4
	g_eps = np.zeros_like(x)
	for i in range(len(x)):
		x1 = x.copy()
		x2 = x.copy()
		x1[i] += eps
		x2[i] -= eps
		g_eps[i] = (cost(x1) - cost(x2))/(2 * eps)
	g_formula = grad(x)
	if np.linalg.norm(g_formula - g_eps) > 1e-5:
		print("WARNING ABOUT GRADIENT FUNCTION!")

def gradient_descent(x_init, learning_rate, iteration):
	x_list = [x_init]
	for i in range(iteration):
		x_new = x_list[-1] - learning_rate*grad(x_list[-1])
		#When to end the GD loop
		if np.linalg.norm(grad(x_new), 2) * 1/len(x_new) < 1:
			break
		x_list.append(x_new)
	return x_list

#Draw points
A = np.array([[2,9,7,9,11,16,25,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T
# b = np.array([np.arange(2,17,1)]).T
fig1 = plt.figure("GD for Linear Regression")
ax = plt.axes(xlim=(-10,60), ylim=(-1,20))
plt.plot(A,b, "ro")

#Draw line by Linear Regression
lr = linear_model.LinearRegression()
lr.fit(A,b)
x0_gd = np.array([np.linspace(1,46,2,dtype=int)]).T
y0_sklearn = lr.coef_*x0_gd + lr.intercept_
plt.plot(x0_gd, y0_sklearn, color='g')

#Add ones to matrix A
ones = np.ones((A.shape[0], 1), dtype=int)
A = np.concatenate((A, ones), axis=1)

#Draw line by Gradient Descent
x_init = np.array([[1.0], [2.0]])
check_grad(x_init)
y0_init = x_init[0][0]*x0_gd + x_init[1][0]
plt.plot(x0_gd,y0_init, color="black")

iteration = 100
learning_rate = 0.0001

x_list = gradient_descent(x_init, learning_rate, iteration)

for i in range(len(x_list)):
	y0_x_list = x_list[i][0] * x0_gd + x_list[i][1]
	plt.plot(x0_gd, y0_x_list, color = "black", alpha=0.5)

#Draw animation
line , = ax.plot([],[], color = "blue")
def update(i):
	y0_gd = x_list[i][0]*x0_gd + x_list[i][1] 
	line.set_data(x0_gd, y0_gd)
	return line,

iters = np.arange(1,len(x_list), 1)
line_ani = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True)

plt.show()

#Draw chart between cost(x) and iteration
# iteration = []
# cost_x = []
# for i in range(len(x_list)):
# 	iteration.append(i+1)
# 	cost_x.append(cost(x_list[i]))
# plt.plot(iteration, cost_x)

# plt.show()