import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random


def add_salt_and_pepper_noise(src, density):
    height, width = src.shape
    dst = np.zeros(shape=src.shape, dtype=np.uint8)
    threshold = 1 - density
    for x in range(height):
        for y in range(width):
            rnd = random.random()
            if rnd < density:
                dst[x][y] = LOWEST_INTENSITY
            elif rnd > threshold:
                dst[x][y] = HIGHEST_INTENSITY
            else:
                dst[x][y] = src[x][y]


    return dst

LOWEST_INTENSITY = 0
HIGHEST_INTENSITY = 255
DENSITY = 0.2

src = cv.imread(filename='../img/Lena.bmp', flags= 0)
#cv.imshow(winname='original', mat= src)

# salt_and_pepper_noised_img_005 = add_salt_and_pepper_noise(src=src, density=DENSITY)
# cv.imwrite(filename='../result_img/salt_and_pepper_noised_img_005.bmp', img=salt_and_pepper_noised_img_005)

# salt_and_pepper_noised_img_01 = add_salt_and_pepper_noise(src=src, density=DENSITY)
# cv.imwrite(filename='../result_img/salt_and_pepper_noised_img_01.bmp', img=salt_and_pepper_noised_img_01)

salt_and_pepper_noised_img_02 = add_salt_and_pepper_noise(src=src, density=DENSITY)
cv.imwrite(filename='../result_img/salt_and_pepper_noised_img_02.bmp', img=salt_and_pepper_noised_img_02)


plt.subplot(121), plt.imshow(X=src, cmap='gray')
# plt.subplot(122), plt.imshow(X=salt_and_pepper_noised_img_005, cmap='gray')
# plt.subplot(122), plt.imshow(X=salt_and_pepper_noised_img_01, cmap='gray')
plt.subplot(122), plt.imshow(X=salt_and_pepper_noised_img_02, cmap='gray')
plt.show()


