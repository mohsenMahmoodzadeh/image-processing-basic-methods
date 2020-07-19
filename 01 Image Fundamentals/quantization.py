import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

src_path = r'.\img\Barbara.bmp'
src = cv.imread(src_path, 0)

cv.imwrite(r'.\img\barbara.bmp', src)

HIGH_INTENSITY = 255



def find_level(index,level):
    i = index
    while True:
        if level[i] != -1:
            level[index] = level[i]
            break
        else:
            i -= 1
    return level[index]


def quantize_image(src, levels=256):
    dst = np.zeros(src.shape, dtype=np.uint8)
    step = math.ceil(HIGH_INTENSITY/ levels)
    level = - np.ones(HIGH_INTENSITY + 1)
    #level[0] = 0
    count = 0
    for i in range(0, HIGH_INTENSITY + 1,step):
        level[i] = count
        count += 1

    for i in range(HIGH_INTENSITY):
        level[i] = find_level(i, level)

    height, width = src.shape

    for x in range(height):
        for y in range(width):
            dst[x, y] = level[src[x, y]]

    return dst


def calcHist(src):

    frequencies = np.zeros((256, 1), dtype=np.uint8)
    height, width = src.shape
    for x in range(height):
        for y in range(width):
            frequencies[src[x, y],0] = frequencies[src[x, y],0] + 1


    plt.hist(frequencies, bins= 256)
    plt.show()


calcHist(src)

cv.equalizeHist()
# dst = quantize_image(src,256)
#
#
#
# cv.imwrite(r'.\img\quantizedBarbara.bmp', dst)
#
# cv.imshow('quantizedBarbara',dst)
# cv.waitKey(0)
# cv.destroyAllWindows()