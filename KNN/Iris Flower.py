import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random import randint

def calculate_accuracy(y_predict,y_test):
	count = 0
	for i in range(len(y_predict)):
		if y_predict[i] == y_test[i]:
			count += 1
	acc = count/len(y_predict)*100
	return acc

def distance(p1,p2):
	d = np.linalg.norm(p1-p2,2)
	return d

def sort(distances,k):
	dis_sort = np.sort(distances)
	return dis_sort[:k]

def max_count(y_list):
	y_predict = []
	iris_types = np.array([0,1,2])
	for y in y_list:
		max_val = 0
		for iris in iris_types:
			count = np.count_nonzero(y==iris)
			if count > max_val:
				max_val = count
				y_temp = iris
		y_predict.append(y_temp)
	return y_predict

def predict(X_predict,X_train,y_train,k):
	y_list = []
	for X1 in X_predict:
		distances = []
		y = []
		for X2 in X_train:
			distances.append(distance(X1, X2))
		dis_sort = sort(distances,k)
		# print(y_train)
		# print(distances)
		# print(dis_sort)
		for dis in dis_sort:
			# print(distances.index(dis))
			y.append(y_train[distances.index(dis)])
		y_list.append(y)
	
	y_predict = np.array(max_count(y_list))
	# print(y_list)
	# print(y_predict)
	return y_predict

def apply_theory(iris,k):

	#shuffle data
	randIndex = np.arange(len(iris.data))
	np.random.shuffle(randIndex)
	iris_X = iris.data[randIndex]
	iris_y = iris.target[randIndex]

	#slice data
	X_train = iris_X[:100]
	X_test = iris_X[100:]
	y_train = iris_y[:100]
	y_test = iris_y[100:]

	y_predict = predict(X_test, X_train, y_train, k)
	print(y_predict)
	print(y_test)
	print(calculate_accuracy(y_predict,y_test))

	k_list = [*range(1,11,1)]
	acc_list = []
	for k in k_list:
		y_predict = predict(X_test, X_train, y_train, k)
		acc_list.append(calculate_accuracy(y_predict, y_test))
	plt.plot(k_list,acc_list, "ro")
	# plt.show()

def use_library(iris,k):
	
	#shuffle data
	randIndex = np.arange(len(iris.data))
	np.random.shuffle(randIndex)
	iris_X = iris.data[randIndex]
	iris_y = iris.target[randIndex]

	X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

	print(X_train[:10])
	print(y_train[:10])
	print(X_train.shape)
	print(y_train.shape)

	knn = neighbors.KNeighborsClassifier(n_neighbors = k)
	knn.fit(X_train, y_train)
	y_predict = knn.predict(X_test)

	accuracy = accuracy_score(y_predict, y_test)
	print(accuracy)

def main():
	iris = datasets.load_iris()
	k=2

	# apply_theory(iris,k)
	use_library(iris,k)

main()