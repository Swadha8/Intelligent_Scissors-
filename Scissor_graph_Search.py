# import cv2 
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# from math import pi
from Intel_Scissor import *

def LUT(n):
	LUT =[[-1,-1], [0, -1], [1, -1], [1,0], [1,1], [1,0], [1, -1], [-1,0]]
	LUT= np.array(LUT)
	return LUT[n]  

def Activate_neighbour(Neighbour_matrix):
	x,y,z  = Neighbour_matrix.shape
	l1 = np.array([0,1,2])
	l2 = np.array([4,5,6])
	l3 = np.array([2,3,4])
	l4 = np.array([0,7,6])

	l1_1 = np.array([0,1,2,6,7])
	l2_1 = np.array([0,1,2,3,4])
	l3_1 = np.array([2,3,4,5,6])
	l4_1 = np.array([0,4,5,6,7])

	for j in range(1, y-1):
		for k in l1:
			Neighbour_matrix[0][j][k] = False
	for j in range(1, y-1):
		for k in l2:
			Neighbour_matrix[x-1][j][k] = False
	for i in range(1, x-1):
		for k in l3:
			Neighbour_matrix[i][y-1][k] = False
	for i in range(1, x-1):
		for k in l4:
			Neighbour_matrix[i][0][k] = False

	for k in l1_1:
		Neighbour_matrix[0][0][k] = False
	for k in l2_1:
		Neighbour_matrix[0][y-1][k] = False
	for k in l3_1:
		Neighbour_matrix[x-1][y-1][k] = False
	for k in l4_1:
		Neighbour_matrix[x-1][0][k] = False

	return Neighbour_matrix


class Node_Build():	
	def __init__(self, image):
		self.image = image
		self.image_cost = Local_cost(image)
		self.image_cost.Laplacian()
		self.f_z = self.image_cost.F_Z()
		self.f_g = self.image_cost.F_G()
		self.weight_z = 0.43
		self.weight_g = 0.14
		self.weight_d = 0.43
		# print('1')
		

	def Neighbour(self):
		x, y  = self.image.shape
		b= 8
		self.Neighbour = np.zeros((x,y,b))
		self.Neighbour_active = np.ones((x,y,b), dtype=bool)

		for j in range (1, y-1):

		for i in range(1,x-1):
			print(i)
			for j in range (1, y-1):
				for k in range (0, 8):
					p,q = LUT(k)
					cost = self.weight_z*self.f_z[i][j] + self.weight_g*self.f_g[i][j]
					if (k%2 != 0):
						self.Neighbour[i][j][k] = cost + self.weight_d*self.image_cost.F_D([i,j], [i+p, j+q])/1.414
					else:
						self.Neighbour[i][j][k] = cost + self.image_cost.F_D([i,j], [i+p, j+q])
		return self.Neighbour

class Graph_search():
	def __init__(self, Neighbour):
		self.Neighbour = Neighbour
		a,b,c = self.Neighbour.shape
		self.visit_matrix = np.zeros((a,b), dtype=bool)
		self.total_cost = np.zeros((a,b))
		print(self.visit_matrix)


image = cv2.imread('/home/swadha/Pictures/images.jpeg',0)
a = Node_Build(image)
b = a.Neighbour()
c = Graph_search(b)
# plt.plot(b)

