import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def median_filter(src, kernel_size):
    temp = []
    indexer = kernel_size // 2
    height = len(src)
    width = len(src[0])
    filtered_img = []
    filtered_img = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > height - 1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > width - 1:
                        temp.append(0)
                    else:
                        for k in range(kernel_size):
                            temp.append(src[i + z - indexer][j + k - indexer])

            temp.sort()
            filtered_img[i][j] = temp[len(temp) // 2]
            temp = []
    return filtered_img


# def median_filter(src,kernel_size):
#     filtered_img = src
#
#     height, width = src.shape
#     indexer = kernel_size // 2
#
#     for i in range(indexer, height-indexer):
#         for j in range(indexer, width-indexer):
#             block = src[i-indexer: i+indexer+1, j-indexer : j+indexer+1]
#             # median = block.flatten().sort()[len(block) // 2]
#             # np.ndarray.flatten(block)
#             temp = block.reshape([1, kernel_size * kernel_size])
#             temp[0].sort()
#             median = temp[0, len(temp[0]) // 2]
#
#             # block.sort()
#             # median = block[indexer, indexer]
#             # filtered_pixel_value = np.median(block)
#             # filtered_img[i][j] = int(filtered_pixel_value)
#             filtered_img[i][j] = median
#     return filtered_img

def immse(A, B):
    return (np.square(A - B)).mean()

# src = cv.imread(filename='../img/salt_and_pepper_noised_img_005.bmp', flags= 0)
# filtered_img_005_3 = median_filter(src=src, kernel_size=3)
# filtered_img_005_5 = median_filter(src=src, kernel_size=5)
# filtered_img_005_7 = median_filter(src=src, kernel_size=7)
# filtered_img_005_9 = median_filter(src=src, kernel_size=9)
# print('immse for density=0.05 and kernel size 3 = ', immse(A=src, B=filtered_img_005_3))
# print('immse for density=0.05 and kernel size 5 = ', immse(A=src, B=filtered_img_005_5))
# print('immse for density=0.05 and kernel size 7 = ', immse(A=src, B=filtered_img_005_7))
# print('immse for density=0.05 and kernel size 9 = ', immse(A=src, B=filtered_img_005_9))

# src = cv.imread(filename='../img/salt_and_pepper_noised_img_01.bmp', flags= 0)
# filtered_img_01_3 = median_filter(src=src, kernel_size=3)
# filtered_img_01_5 = median_filter(src=src, kernel_size=5)
# filtered_img_01_7 = median_filter(src=src, kernel_size=7)
# filtered_img_01_9 = median_filter(src=src, kernel_size=9)
# print('immse for density=0.1 and kernel size 3 = ', immse(A=src, B=filtered_img_01_3))
# print('immse for density=0.1 and kernel size 5 = ', immse(A=src, B=filtered_img_01_5))
# print('immse for density=0.1 and kernel size 7 = ', immse(A=src, B=filtered_img_01_7))
# print('immse for density=0.1 and kernel size 9 = ', immse(A=src, B=filtered_img_01_9))

src = cv.imread(filename='../img/salt_and_pepper_noised_img_02.bmp', flags= 0)
filtered_img_02_3 = median_filter(src=src, kernel_size=3)
filtered_img_02_5 = median_filter(src=src, kernel_size=5)
filtered_img_02_7 = median_filter(src=src, kernel_size=7)
filtered_img_02_9 = median_filter(src=src, kernel_size=9)
print('immse for density=0.2 and kernel size 3 = ', immse(A=src, B=filtered_img_02_3))
print('immse for density=0.2 and kernel size 5 = ', immse(A=src, B=filtered_img_02_5))
print('immse for density=0.2 and kernel size 7 = ', immse(A=src, B=filtered_img_02_7))
print('immse for density=0.2 and kernel size 9 = ', immse(A=src, B=filtered_img_02_9))

cv.imwrite(filename='../result_img/filtered_img_02_9.bmp', img=filtered_img_02_9)
plt.subplot(121), plt.imshow(X=src, cmap='gray')
plt.subplot(122), plt.imshow(X=filtered_img_02_9, cmap='gray')
plt.show()
# cv.imshow(winname='salt_and_pepper_noised_img', mat=salt_and_pepper_noised_img)

# cv.imshow('filtered', filtered_img_005_3)
# cv.waitKey(0)