import pygame
from random import randint
import math
from sklearn.cluster import KMeans

def distance(p1, p2):
	return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

pygame.init()
screen = pygame.display.set_mode((1000, 600))
pygame.display.set_caption('Kmeans Distribution')
running = True
BLACK = (0,0,0)
GRAY = (128,128,128)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (147,153,35)
PURPLE = (255,0,255)
SKY = (0,255,255)
ORANGE = (255,125,25)
GRAPE = (100,25,125)
GRASS = (55,155,65)
COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE, SKY, ORANGE, GRAPE, GRASS]
K = 0
error = 0
points = []
clusters = []
labels = []
font = pygame.font.SysFont('Sans', 30)
font_small = pygame.font.SysFont('Sans', 20)
plus_text = font.render('+', True, WHITE)
minus_text = font.render('-', True, WHITE)
run_text = font.render('Run', True, WHITE)
random_text = font.render('Random', True, WHITE)
algorithm_text = font.render('Algorithm', True, WHITE)
reset_text = font.render('Reset', True, WHITE)
clock = pygame.time.Clock()

while running:		
	clock.tick(60)
	screen.fill(GRAY)
	x, y = pygame.mouse.get_pos()
	k_text = font.render('K = ' + str(K), True, BLACK)

	#draw screen
	pygame.draw.rect(screen, BLACK, (45,45,610,410))
	pygame.draw.rect(screen, WHITE, (50,50,600,400))

	#display(x, y) position
	xy_pos_text = font_small.render(str(x - 50) + ', ' + str(y - 50), True, BLACK)
	if 50 <= x <= 650 and 50 <= y <= 450:
		screen.blit(xy_pos_text, (x + 15, y))

	#draw plus and minus buttons
	pygame.draw.rect(screen, BLACK, (700,50,50,50))
	pygame.draw.rect(screen, BLACK, (800,50,50,50))
	screen.blit(plus_text, (715,55))
	screen.blit(minus_text, (815,55))

	#draw K
	screen.blit(k_text, (900,55))

	#draw run button
	pygame.draw.rect(screen, BLACK, (700,150,150,50))
	screen.blit(run_text, (750,155))

	#draw random button
	pygame.draw.rect(screen, BLACK, (700,250,150,50))
	screen.blit(random_text, (725,255))

	#draw algorithm button
	pygame.draw.rect(screen, BLACK, (700,400,150,50))
	screen.blit(algorithm_text, (725,405))

	#draw reset button
	pygame.draw.rect(screen, BLACK, (700,500,150,50))
	screen.blit(reset_text, (725,505))

	for event in pygame.event.get():
		if event.type == pygame.MOUSEBUTTONDOWN:
			#create points
			if 50 <= x <= 650 and 50 <= y <= 450:
				labels = []
				point = (x, y)
				points.append(point)
				print(points)
			#add and subtract K value
			if 700 <= x <= 750 and 50 <= y <= 100:
				if K < 9:
					K += 1
			if 800 <= x <= 850 and 50 <= y <= 100:
				if K > 0:
					K -= 1
			#random button click
			if 700 <= x <= 850 and 250 <= y <= 300:
				#create clusters
				clusters = []
				labels = []
				for i in range(K):
					random_point = (randint(50, 650), randint(50, 450))
					clusters.append(random_point)
			#run button click
			if 700 <= x <= 850 and 150 <= y <= 200:
				if clusters == []:
					continue
				#Assign labels
				labels = []
				for point in points:
					distances_to_cluster = []
					for cluster in clusters:
						dis = distance(point, cluster)
						distances_to_cluster.append(dis)
					labels.append(distances_to_cluster.index(min(distances_to_cluster)))
				#Center clusters
				for i in range(K):
					sum_x = 0
					sum_y = 0
					count = 0
					for j in range(len(points)):
						if labels[j] == i:
							sum_x += points[j][0]
							sum_y += points[j][1]
							count += 1
					if count != 0:
						new_cluster_x = sum_x/count
						new_cluster_y = sum_y/count
						clusters[i] = [new_cluster_x, new_cluster_y]
			#algorithm button click
			if 700 <= x <= 850 and 400 <= y <= 450:
				if K > 0 and points != []:
					kmeans = KMeans(n_clusters=K).fit(points)
					labels = kmeans.predict(points)
					clusters = kmeans.cluster_centers_
			#reset button click
			if 700 <= x <= 850 and 500 <= y <= 550:
				K = 0
				points = []
				clusters = []
				labels = []
				error = 0
		if event.type == pygame.QUIT:
			running = False

	#draw clusters
	for i in range(len(clusters)):
		pygame.draw.circle(screen, COLORS[i], (int(clusters[i][0]), int(clusters[i][1])), 10)

	#draw points
	for i in range(len(points)):
		if labels == []:
			pygame.draw.circle(screen, BLACK, (points[i][0], points[i][1]), 6)
			pygame.draw.circle(screen, WHITE, (points[i][0], points[i][1]), 5)
		else:
			pygame.draw.circle(screen, COLORS[labels[i]], (points[i][0], points[i][1]), 5)

	#calculate error
	error = 0
	if labels != []:
		for i in range(len(points)):
			error += distance(points[i], clusters[labels[i]])
	
	#draw Error
	error_text = font.render('Error = ' + str(int(error)), True, BLACK)
	screen.blit(error_text, (700,330))

	pygame.display.flip()

pygame.quit()