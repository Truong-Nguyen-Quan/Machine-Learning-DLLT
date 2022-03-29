import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.animation as animation

def main():
	A = np.array([np.arange(2,26)]).T
	b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T

	fig1 = plt.figure("Draw parabol from data by GD")
	ax = plt.axes(xlim=(-10,50), ylim=(-10,50))
	plt.title("Draw parabol by Gradient Descent")
	plt.plot(A,b, 'ro')

	A_square = A*A
	ones = np.ones((A.shape[0], 1), dtype=int)
	A = np.concatenate((A_square, A, ones), axis=1)

	draw_parabol_lib(A,b)
	draw_parabol_math(A,b)
	draw_parabol_gd(A,b,fig1,ax)
	plt.show()

#Draw parabol by using sklearn library
def draw_parabol_lib(A,b):
	lr = linear_model.LinearRegression()
	lr.fit(A,b)
	x0 = np.array([np.arange(1,30)]).T
	y0_sklearn = lr.coef_[0][0]*x0*x0 + lr.coef_[0][1]*x0 + lr.intercept_
	plt.plot(x0, y0_sklearn, color='g')

#Draw parabol by using math
def draw_parabol_math(A,b):
	x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
	x0 = np.arange(30)
	y0 = x[0][0]*(x0*x0) + x[1][0]*x0 + x[2][0]
	plt.plot(x0,y0)

#Draw parabol by using gradient descent
#Return a value
def cost(A,b,x):
	m = A.shape[0]
	return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2

#Return a matrix
def grad(A,b,x):
	m = A.shape[0]
	return 1/m * A.T.dot(A.dot(x) - b)

#Return x
def gradient_descent(A, b, x_init, learning_rate, iteration):
	x_list = [x_init]
	for i in range(iteration):
		x_new = x_list[-1] - learning_rate*grad(A,b,x_list[-1])
		#When to end the GD loop
		if np.linalg.norm(grad(A,b,x_new), 2) * 1/len(x_new) < 1:
			break
		x_list.append(x_new)
	return x_list

#Check grad formula
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

def make_animation(x_list, x0_gd, fig1, ax):
	line , = ax.plot([],[], color = "blue")
	def update(i):
		y0_ani = x_list[i][0]*(x0_gd*x0_gd) + x_list[i][1]*x0_gd + x_list[i][2] 
		line.set_data(x0_gd, y0_ani)
		return line,

	iters = np.arange(1,len(x_list), 1)
	line_ani = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True)
	plt.show()

def draw_parabol_gd(A,b,fig1,ax):
	#Draw initial parabol
	x_init = np.array([[-0.5], [8.0], [-20.0]])
	x0_gd = np.array([np.arange(1,30)]).T
	check_grad(A,b,x_init)
	y0_init = x_init[0][0]*(x0_gd*x0_gd) + x_init[1][0]*x0_gd + x_init[2][0]
	plt.plot(x0_gd,y0_init, color="black")

	#Choose learning rate and iteration
	iteration = 80
	learning_rate = 0.000001

	#Calculate
	x_list = gradient_descent(A, b, x_init, learning_rate, iteration)

	#Draw parabol
	for i in range(len(x_list)):
		y0_x_list = x_list[i][0]*(x0_gd*x0_gd) + x_list[i][1]*x0_gd + x_list[i][2]
		plt.plot(x0_gd, y0_x_list, color = "black", alpha=0.5)

	#Add animation
	make_animation(x_list,x0_gd,fig1,ax)

	#Draw the graph between cost(x) and iteration
	y0_costx = []
	for i in range(len(x_list)):
		y0_costx.append(cost(A,b,x_list[i]))
	plt.plot(np.arange(1,len(y0_costx)+1), y0_costx)

main()