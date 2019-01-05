import cv2 
import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi

def Normalise(vec):
	vec = np.array(vec)
	sum = 0.0
	for i in range (0, len(vec)):
		sum += vec[i]*vec[i]
	if (sum == 0):
		return vec
	else:
		vec = vec/sum
		return vec


class Local_cost():
	def __init__(self, image):
		self.image = image

	def Laplacian(self):
		## Calculate Image features
		self.f_x = cv2.Sobel(self.image,cv2.CV_64F,1,0,ksize=3)
		self.f_y = cv2.Sobel(self.image,cv2.CV_64F,0,1,ksize=3)
		self.laplacian = cv2.Laplacian(self.image,cv2.CV_64F)
		return self.laplacian

	def F_Z(self):
		self.f_z = np.zeros(self.image.shape)
		for i,j in np.argwhere(self.laplacian == 0):
			self.f_z[i][j] = 1
		return self.f_z

	def F_G(self):

		self.Grad_matrix = np.sqrt(np.multiply(self.f_x,self.f_x) + np.multiply(self.f_y,self.f_y))
		self.G_max = np.amax(self.Grad_matrix)
		self.f_g = np.ones(self.image.shape) - self.Grad_matrix/self.G_max

		return self.f_g

	def F_D(self, p, q):  
	# p, q are position vectors in the image
		self.D_P = (self.f_y[p[0]][p[1]], (-1)*self.f_x[p[0]][p[1]]) 
		self.D_Q = (self.f_y[q[0]][q[1]], (-1)*self.f_x[q[0]][q[1]])

		if ( self.D_P[0]*(p[0]-q[0]) + self.D_P[1]*(p[1]-q[1]) >= 0 ):
			self.L_P_Q = (q[0]-p[0], q[1]-p[1])
		else:
			self.L_P_Q = (p[0]-q[0], p[1]-q[1])

		self.D_P = Normalise(self.D_P)
		self.D_Q = Normalise(self.D_Q)
		self.L_P_Q = Normalise(self.L_P_Q)		

		d_p_pq = self.D_P[0]*self.L_P_Q[0] + self.D_P[1]*self.L_P_Q[1]
		d_q_pq = self.D_Q[0]*self.L_P_Q[0] + self.D_Q[1]*self.L_P_Q[1]

		f_d = (math.acos(d_p_pq) + math.acos(d_q_pq))/pi
		return f_d


# plt.subplot(2,2,1),plt.imshow(image, cmap='gray')
# plt.title('Original')
# plt.subplot(2,2,2),plt.imshow(a.f_x, cmap='gray')
# plt.title('Laplacian')
# plt.subplot(2,2,3),plt.imshow(a.f_y, cmap='gray')
# plt.title('Sobel_x')

# plt.subplot(2,2,4),plt.imshow(b, cmap='gray')
# plt.title('Sobel_y')

# plt.show()



