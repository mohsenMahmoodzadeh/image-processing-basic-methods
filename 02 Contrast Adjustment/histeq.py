import numpy as np
import matplotlib.pyplot as plt
import cv2
'''
1. Find frequency of each pixel.
2. Find the probability density fucntion of each frequency.
3. Find the cumulative histogram of each pixel.
4. Find the cumulative distribution probability of each pixel.
5. Calculating final new pixel value by cdf * (no of bits).
6. Replace it with oroginal values.
'''

img = cv2.imread('./img/Camera Man.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src',img)
cv2.waitKey(0)


plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')



#To display image before equalization




a = np.zeros((256,),dtype=np.float16)
b = np.zeros((256,),dtype=np.float16)

height,width=img.shape

#finding histogram
for i in range(width):
    for j in range(height):
        g = img[j,i]
        a[g] = a[g]+1

#print(a)
plt.subplot(2,2,2)
plt.plot(a)

#performing histogram equalization
tmp = 1.0/(height*width)
b = np.zeros((256,),dtype=np.float16)

for i in range(256):
    for j in range(i+1):
        b[i] += a[j] * tmp;
    b[i] = round(b[i] * 255);

# b now contains the equalized histogram
b=b.astype(np.uint8)
plt.subplot(2,2,4)
plt.plot(b)
#print(b)

#Re-map values from equalized histogram into the image
for i in range(width):
    for j in range(height):
        g = img[j,i]
        img[j,i]= b[g]


plt.subplot(2,2,3)
plt.imshow(img, cmap='gray')

cv2.imshow('dest',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.show()
def histeq(img):
    height, width = img.shape
    intensities = np.linspace(start=0, stop=255, num=256, endpoint=True, dtype=int)
    img = img.flatten()

    # print(len(img)) # 262144
    frequencies = np.zeros(256, dtype=np.uint8)
    # frequencies = np.zeros(256, dtype=np.uint8)
    print(frequencies)
    for pixel in range(len(img)):
        frequencies[img[pixel]] = frequencies[img[pixel]] + 1
    print(frequencies)