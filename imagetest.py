import pdb
import imageio                     #imread(), imwrite()
import matplotlib.pyplot as plt
import numpy as np                 #array(), zeros()
from random import *               #random()
from scipy.spatial import Delaunay #Delaunay()

#getting weird stack overflow in getAVGColor when using the hummingbird image
imgin = imageio.imread('large.png')
imgout = np.zeros((imgin.shape[0], imgin.shape[1], 3), dtype = np.uint8)
#plt.imshow(imgin)
points = np.empty((0,2), int)

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

#@isTouched: Used to prevent calculating a value twice
#@AVGColor: an updating average of RGB values within the triangle @tri
#@tri: [[x,y],[x,y],[x,y]] 3 points that define a triangle
#@numpts: used to calculate the average
#@point: [x,y] point on imgin
#
#Traverses through points within @tri by recursing through the top, left, right
# and bottom pixel relative to the current @point.
isTouched = np.empty((0,2), int)
AVGColor = np.zeros((3,), float)
numpts = 0
def getAVGColor(point, tri):
	global isTouched
	global AVGColor
	global numpts
	global imgin  #readonly
	global imgout #readonly

	#if current point is out of bounds
	if (point[0] < 0) or (point[0] > imgin.shape[0]) or\
		(point[1] < 0) or (point[1] > imgin.shape[1]):
		return
	#if we've already seen this point
#https://stackoverflow.com/questions/25823608/find-matching-rows-in-2-dimensional-numpy-array
	if len((isTouched == point).all(axis = 1).nonzero()[0]) > 0:
		return
	#if @point is not within triangle @tri
	if(not isInTri(point, tri[0], tri[1], tri[2])):
		return
		
	#add point to known points
	isTouched = np.append(isTouched, [point], axis = 0)
	#add current points's RGB values to rolling average
	AVGColor[0] = (numpts * AVGColor[0] + imgin[point[1], point[0], 0])/float(numpts+1)
	AVGColor[1] = (numpts * AVGColor[1] + imgin[point[1], point[0], 1])/float(numpts+1)
	AVGColor[2] = (numpts * AVGColor[2] + imgin[point[1], point[0], 2])/float(numpts+1)
	numpts += 1
	#pdb.set_trace()

	#recurse through points top, down, left, right of current point
	getAVGColor([point[0]-1,point[1]], tri)
	getAVGColor([point[0]+1,point[1]], tri)
	getAVGColor([point[0],point[1]-1], tri)
	getAVGColor([point[0],point[1]+1], tri)
	return

#@point: [x,y] array of a point in polygon to rasterize
#@tri: [[x,y],[x,y],[x,y]] array of three points defining a triangle
#@color: [R,G,B] array of three uint8 values describing the color to fill
def rasterize(point, tri):
	global imgout
	global AVGColor

	#if we've already seen this point
	if(imgout[point[1],point[0], 0] != 0):
		return
	#if point is not within triangle defined by tri
	if(not isInTri(point, tri[0], tri[1], tri[2])):
		return
	#color the pixel
	imgout[point[1],point[0], 0] = AVGColor[0]
	imgout[point[1],point[0], 1] = AVGColor[1]
	imgout[point[1],point[0], 2] = AVGColor[2]
	#recurse through points top, down, left, right of current point
	rasterize([point[0]-1,point[1]], tri)
	rasterize([point[0]+1,point[1]], tri)
	rasterize([point[0],point[1]-1], tri)
	rasterize([point[0],point[1]+1], tri)
	return

#img.shape = (1603, 2403, 3)
#points = [[x,y]...]
#random points on the image
for i in range(imgin.shape[1]):
	for j in range(imgin.shape[0]):
		if random() < .001:
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
	getAVGColor(centroid, points[tris.simplices[i]])
	rasterize(centroid, points[tris.simplices[i]])
	rasterize(points[tris.simplices[i,0]], points[tris.simplices[i]])
	rasterize(points[tris.simplices[i,1]], points[tris.simplices[i]])
	rasterize(points[tris.simplices[i,2]], points[tris.simplices[i]])

plt.triplot(points[:,0], points[:,1], tris.simplices.copy())
plt.imshow(imgout)
imageio.imwrite('medium_test.png', imgout)
#plt.plot(points[:,0], points[:,1], 'o')
plt.show()
