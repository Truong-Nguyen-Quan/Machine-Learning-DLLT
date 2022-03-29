import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy

img = plt.imread('flower.jpg')
width = img.shape[0]
height = img.shape[1]
print(img[0][0])

# list_for_img2 = []
# for i in range(len(img)):
# 	for j in range(len(img[i])):
# 		list_for_img2.append(img[i][j])
# img2 = numpy.array(list_for_img2)
# print(img2.shape)

img = img.reshape(width * height, 3)
print(img[0])
print(img.shape)

kmeans = KMeans(n_clusters=5).fit(img)
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_
# print(clusters)
# print(labels)

img2 = numpy.zeros_like(img)
# img2 = numpy.zeros((width, height, 3), dtype = numpy.uint8)
# index = 0
# for i in range(width):
# 	for j in range(height):
# 		img2[i][j] = clusters[labels[index]]
# 		index += 1

for i in range(len(img2)):
	img2[i] = clusters[labels[i]]

img2 = img2.reshape(width, height, 3)

plt.imshow(img2)
plt.show()