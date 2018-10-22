import pdb
import imageio                     #imread(), imwrite()
import matplotlib.pyplot as plt
import numpy as np                 #array(), zeros()
from random import *               #random()
from scipy.spatial import Delaunay #Delaunay()
import queue                       #queue()

imgin = imageio.imread('Hummingbird')
imgout = np.zeros((imgin.shape[0], imgin.shape[1], 3), dtype = np.uint8)
points = np.empty((0,2), int)
class Pixel:
	def __init__(self, x, y, color):
		self.coor = (x,y)
		self.x = x
		self.y = y
		self.colorR = color[0]
		self.colorG = color[1]
		self.colorB = color[2]
	def __eq__(self, other):
		return self.x == other.x and self.y == other.y
	def __str__(self):
		return "[%s,%s] [%s, %s, %s]\n" % (self.x, self.y, self.colorR, self.colorG, self.colorB)
	def __repr__(self):
		return "[%s,%s] [%s, %s, %s]\n" % (self.x, self.y, self.colorR, self.colorG, self.colorB)
#@tri:[[x,y],[x,y],[x,y]] array that defines a triangle
#return: [x,y] centroid for @tri
def getCentroid(tri):
	centX = int((tri[0][0] + tri[1][0] + tri[2][0])/3)
	centY = int((tri[0][1] + tri[1][1] + tri[2][1])/3)
	return np.array([centX, centY])

#https://www.gamedev.net/forums/topic/295943-is-this-a-better-point-in-triangle-test-2d/
def sign(p1, p2, p3):
	#return (p1[1] - p3[1])*(p2[0] - p3[0]) - (p2[1] - p3[1])*(p1[0] - p3[0])
	return ((p1[0] - p3[0])*(p2[1] - p3[1])) - ((p2[0] - p3[0])*(p1[1] - p3[1]))

def isInTri(point, tri1, tri2, tri3):
	b1 = sign(point, tri1, tri2) < 0
	b2 = sign(point, tri2, tri3) < 0
	b3 = sign(point, tri3, tri1) < 0
	return ((b1 == b2) and (b2 == b3))

#@tri: [[x,y],[x,y],[x,y]] 3 points that define a triangle
#@point: [x,y] point on imgin
#
#Instead of recursing, this algorithm pushes Pixel objects to the queue @pixQ
#
#return: Pixel object containing the average RGB of @tri
touched = np.zeros((imgin.shape[0], imgin.shape[1]), dtype = np.bool_)
def getAVGColor(point, tri):
	global imgin
	global touched
	AVGColor = Pixel(point[0], point[1], imgin[point[1], point[0], :])
	if(not isInTri(point, tri[0], tri[1], tri[2])):
		return AVGColor
	numpts = 0
	pixQ = queue.Queue(0)
	currPix = Pixel(point[0], point[1], imgin[point[1], point[0], :])
	pixQ.put(currPix)
	i = 0
	while(not pixQ.empty()):
		currPix = pixQ.get()
		if(touched[currPix.y, currPix.x]):
			continue
			
		touched[currPix.y, currPix.x] = True

		#update the average
		AVGColor.colorR = (numpts * AVGColor.colorR + currPix.colorR)/(numpts+1)
		AVGColor.colorG = (numpts * AVGColor.colorG + currPix.colorG)/(numpts+1)
		AVGColor.colorB = (numpts * AVGColor.colorB + currPix.colorB)/(numpts+1)
		numpts += 1
		#print(currPix.coor)
		#if(i > 30000):
		#	pdb.set_trace()
		if(isInTri([currPix.x-1, currPix.y], tri[0], tri[1], tri[2])):
			west = Pixel(currPix.x-1, currPix.y, imgin[currPix.y, currPix.x-1 ,:])
			pixQ.put(west)
		if(isInTri([currPix.x+1, currPix.y], tri[0], tri[1], tri[2])):
			east = Pixel(currPix.x+1, currPix.y, imgin[currPix.y, currPix.x+1,:])
			pixQ.put(east)
		if(isInTri([currPix.x, currPix.y-1], tri[0], tri[1], tri[2])):
			south = Pixel(currPix.x, currPix.y-1, imgin[currPix.y-1, currPix.x,:])
			pixQ.put(south)
		if(isInTri([currPix.x, currPix.y+1], tri[0], tri[1], tri[2])):
			north = Pixel(currPix.x, currPix.y+1, imgin[currPix.y-1, currPix.x, :])
			pixQ.put(north)
		i += 1

	return AVGColor

#@pix: Pixel object containing the starting position and color
#@tri: [[x,y],[x,y],[x,y]] array of three points defining a triangle
#
#Uses the same algorithm as getAVGColor() to rasterize a triangle @tri
def rasterize(pix, tri):
	global imgout
	#if point is out of bounds
	if(not isInTri(pix.coor, tri[0], tri[1], tri[2])):
		return
	#if point has already been rasterized
	if(imgout[pix.y, pix.x, 0] != 0):
		return
	pixQ = queue.Queue(0)
	pixQ.put(pix)
	while(not pixQ.empty()):
		currPix = pixQ.get()
		if(imgout[currPix.y, currPix.x, 0] != 0):
			continue
		#rasterize current pixel
		imgout[currPix.y, currPix.x, 0] = pix.colorR
		imgout[currPix.y, currPix.x, 1] = pix.colorG
		imgout[currPix.y, currPix.x, 2] = pix.colorB
		
		if(isInTri([currPix.x-1, currPix.y], tri[0], tri[1], tri[2])):
			west = Pixel(currPix.x-1, currPix.y, [0,0,0])
			pixQ.put(west)
		if(isInTri([currPix.x+1, currPix.y], tri[0], tri[1], tri[2])):
			east = Pixel(currPix.x+1, currPix.y, [0,0,0])
			pixQ.put(east)
		if(isInTri([currPix.x, currPix.y-1], tri[0], tri[1], tri[2])):
			south = Pixel(currPix.x, currPix.y-1, [0,0,0])
			pixQ.put(south)
		if(isInTri([currPix.x, currPix.y+1], tri[0], tri[1], tri[2])):
			north = Pixel(currPix.x, currPix.y+1, [0,0,0])
			pixQ.put(north)

	return

#img.shape = (1603, 2403, 3)
#points = [[x,y]...]
#random points on the image
for i in range(imgin.shape[1]):
	for j in range(imgin.shape[0]):
		if random() < .0001:
			points = np.append(points, [[i,j]], axis = 0)
#points = np.array([[5,38],[17,98],[23, 94],[26,74],[28,51],[43,71],[52,60],\
#[57,37],[60,97],[65,76],[90,56],[91,85]])
#points = np.array([[10,10], [10,70], [70,10]])

#triangulate the points using the Delaunay algorithm
tris= Delaunay(points)
#rasterize the triangulated image
for i in range(tris.simplices.shape[0]):
	isTouched = np.empty((0,2), int)
	AVGColor = np.zeros((3,), float)
	numpts = 0
	centroid = getCentroid(points[tris.simplices[i]])
	print("tri number: %s" % i)
	print("in AVGColor...")
	cent = getAVGColor(centroid, points[tris.simplices[i]])
	print("rasterizing")
	rasterize(cent, points[tris.simplices[i]])

plt.triplot(points[:,0], points[:,1], tris.simplices.copy())
plt.imshow(imgout)
imageio.imwrite('largetest.png', imgout)
#plt.plot(points[:,0], points[:,1], 'o')
plt.show()
