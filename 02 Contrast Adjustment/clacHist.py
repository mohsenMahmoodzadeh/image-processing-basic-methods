import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

'''reading source image'''
img = cv.imread('./img/Camera Man.bmp', 0)

# cv.imshow('cm',img)
# cv.waitKey(0)
# fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2)


# hist = cv.calcHist([img], [0], None, [256], [0, 256])

# plt.hist(img, bins= 255)
# plt.show()

def calcHist(src):

    '''this function computes the histogram of
    a gray-scale image.'''

    height, width = src.shape

    intensities = np.linspace(start=0, stop=255, num=256, endpoint=True, dtype=int)
    img = src.flatten()

    # print(len(img)) # 262144
    frequencies = np.zeros(256, dtype=np.float16)
    # frequencies = np.zeros(256, dtype=np.uint8)
    print(frequencies)
    for pixel in range(len(img)):
        temp = img[pixel]
        frequencies[temp] = frequencies[temp] + 1
        # frequencies[img[pixel]] = frequencies[img[pixel]] + 1
    print(frequencies)

    # for intensity in range(0,256):
    #     for x in range(height):
    #         for y in range(width):
    #             if src[x, y] == intensity:
    #                 frequencies[intensity] += 1

    # print(np.sum(frequencies)) # 28928
    plt.plot(intensities,frequencies.ravel())
    plt.show()

    # plt.hist(img, bins=256, range=[0,256])
    # plt.hist(src.reshape(-1), bins=256)



# plt.hist(img.ravel(),256,[0,256])
# plt.show()

# cv.imwrite('./result_img/Camera Man Hist plt', result)

calcHist(img)