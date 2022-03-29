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
	acc = count/len(y_predict)
	return round(acc,2)

def distance(p1,p2):
	d = np.linalg.norm(p1-p2,2)
	return d

def max_count(y_list):
	y_predict = []
	for i in range(len(y_list)):
		unique_nums = []
		counts = []
		max_number = 0
		#create a list with unique numbers
		for j in range(len(y_list[i])):
			if y_list[i][j] not in unique_nums:
				unique_nums.append(y_list[i][j])
		#count similar numbers in y_list[i]
		for n in unique_nums:
			count = 0
			for y in y_list[i]:
				if y==n:
					count += 1
			counts.append(count)
		#find max number in counts
		for c in counts:
			if c > max_number:
				max_number = c
		y_predict.append(unique_nums[counts.index(max_number)])
	return y_predict

def predict(X_test,X_train,y_train,k):
	y_list = []
	for X1 in X_test:
		distances = []
		y = []
		for X2 in X_train:
			distances.append(distance(X1, X2))
		dis_sort = np.sort(distances)[:k]
		for dis in dis_sort:
			y.append(y_train[distances.index(dis)])
		y_list.append(y)
	
	y_predict = np.array(max_count(y_list))
	return y_predict

def apply_theory(digit,k):

	#shuffle data
	randIndex = np.arange(len(digit.target))
	np.random.shuffle(randIndex)
	digit_X = digit.data[randIndex]
	digit_y = digit.target[randIndex]

	#slice data
	X_train = digit_X[:1437]
	X_test = digit_X[1437:]
	y_train = digit_y[:1437]
	y_test = digit_y[1437:]

	y_predict = predict(X_test, X_train, y_train, k)
	print(calculate_accuracy(y_predict,y_test))

	# k_list = [*range(1,11,1)]
	# acc_list = []
	# for k in k_list:
	# 	y_predict = predict(X_test, X_train, y_train, k)
	# 	acc_list.append(calculate_accuracy(y_predict, y_test))
	# plt.plot(k_list,acc_list, "ro")
	# plt.show()

def use_library(digit,k):
	
	#shuffle data
	randIndex = np.arange(len(digit.target))
	np.random.shuffle(randIndex)
	digit_X = digit.data[randIndex]
	digit_y = digit.target[randIndex]

	X_train, X_test, y_train, y_test = train_test_split(digit_X, digit_y, test_size=50)

	knn = neighbors.KNeighborsClassifier(n_neighbors = k)
	knn.fit(X_train, y_train)
	y_predict = knn.predict(X_test)

	img = X_test[0].reshape(8,8)
	# print(knn.predict(X_test[0].reshape(1, -1)))
	plt.gray()
	plt.imshow(img)
	# plt.show()

	accuracy = accuracy_score(y_predict, y_test)
	print(accuracy)

	return y_predict

def main():
	digit = datasets.load_digits()
	k=2

	#shuffle data
	randIndex = np.arange(len(digit.target))
	np.random.shuffle(randIndex)
	digit_X = digit.data[randIndex]
	digit_y = digit.target[randIndex]

	# apply_theory(digit,k)
	use_library(digit,k)

main()