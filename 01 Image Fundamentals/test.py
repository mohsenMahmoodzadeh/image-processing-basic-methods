import numpy as np
import cv2
import math as m
import sys

path = r'.\img\Elaine.bmp'

img = cv2.imread(path)
angle = 30

#get rotation matrix
def getRMat(cx=0, cy=0, angle=0, scale=1):
    a = scale*m.cos(angle*np.pi/180)
    b = scale*(m.sin(angle*np.pi/180))
    u = (1-a)*cx-b*cy
    v = b*cx+(1-a)*cy
    return np.array([[a,b,u], [-b,a,v]])

#determine shape of img
h, w = img.shape[:2]
#print h, w
#determine center of image
cx, cy = (w / 2, h / 2)

#calculate rotation matrix
#then grab sine and cosine of the matrix
mat = getRMat(cx,cy, -int(angle), 1)
print (mat)
cos = np.abs(mat[0,0])
sin  = np.abs(mat[0,1])

#calculate new height and width to account for rotation
newWidth = int((h * sin) + (w * cos))
newHeight = int((h * cos) + (w * sin))
#print newWidth, newHeight

mat[0,2] += (newWidth / 2) - cx
mat[1,2] += (newHeight / 2) - cy

#this is how the image SHOULD look
dst = cv2.warpAffine(img, mat, (newWidth, newHeight))

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#apply transform
#attempt at my own warp affine function...still buggy tho
def warpAff(image, matrix, width, height):
    dst = np.zeros((width, height, 3), dtype=np.uint8)
    oldh, oldw = image.shape[:2]
    #print oldh, oldw
    #loop through old img and transform its coords
    for x in range(oldh):
        for y in range(oldw):
            #print y, x
            #transform the coordinates
            u = int(x*matrix[0,0]+y*matrix[0,1]+matrix[0,2])
            v = int(x*matrix[1,0]+y*matrix[1,1]+matrix[1,2])
            #print u, v
            #v -= width / 1.5
            if (u >= 0 and u < height) and (v >= 0 and v < width):
                dst[u,v] = image[x,y]
    return dst


dst = cv2.warpAffine(img, mat, (newWidth, newHeight))

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()