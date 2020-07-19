import cv2 as cv
import numpy as np
import statistics
import matplotlib.pyplot as plt
from display import disp_pyr_horizontal


def averaging_filter(src):
    height, width = src.shape
    dst = np.zeros(shape=(height//2, width//2),dtype=np.uint8)
    for x in range(0,height,2):
        for y in range(0, width, 2):
            dst[x//2, y//2] = statistics.mean([src[x, y] , src[x, y+1] , src[x+1, y] , src[x+1, y+1]])
            # dst[x//2, y//2] = (src[x, y] + src[x, y+1] + src[x+1, y] + src[x+1, y+1])/4
    return dst

def interpolation_filter(src):
    height, width = src.shape
    dst = np.zeros(shape=(height * 2, width * 2), dtype=np.uint8)
    i = 0
    j = 0
    for x in range(height):
        for y in range(width):
            pixel = src[x, y]
            dst[x*2, y*2] = pixel
            dst[x*2, y*2+1] = pixel
            dst[x*2+1, y*2] = pixel
            dst[x*2+1, y*2+1] = pixel
    # for x in range(height):
    #     if i >= 2 * height:
    #         i = 0
    #     for y in range(width):
    #         if j >= 2 * width:
    #             j = 0
    #         pixel = src[x, y]
    #         dst[i, j] = pixel
    #         dst[i, j+1] = pixel
    #         dst[i+1, j] = pixel
    #         dst[i+1, j+1] = pixel
    #         j+= 2
    #
    #     i += 2


    return dst

# def disp_fmt_pyr(pyr):
#     """
#     Visualize the pyramid
#     """
#     num_levels = len(pyr)
#
#     H, W = pyr[0].shape
#
#     img_heights = [H * 2 ** (-i) for i in np.arange(num_levels, dtype=float)]
#     H = np.int(np.sum(img_heights))
#
#     out = np.zeros((H, W))
#
#     for i in np.arange(num_levels):
#         rstart = np.int(np.sum(img_heights[:i]))
#         rend = np.int(rstart + img_heights[i])
#
#         out[rstart:rend, :np.int(img_heights[i])] = pyr[i]
#
#     return out

src = cv.imread('../img/Lena.bmp', 0)
# print(src.shape) #(512, 512)
approx_pyr = [src]
prediction_pyr = []
approximation = src
for i in range(4):
    prev_approx = approximation
    approximation = averaging_filter(approximation)
    if i != 3:
        approx_pyr.append(approximation)

    supersampled_approx = interpolation_filter(approximation)
    # print(supersampled_approx.shape)
    prediction = prev_approx - supersampled_approx

    prediction_pyr.append(prediction)


# approx_img = disp_pyr_vertical(approx_pyr)
# approx_img = disp_pyr_horizontal(approx_pyr)
# plt.imshow(approx_img,cmap='gray')
# plt.show()
# cv.imwrite(filename='../result_img/approx_pyramid.png', img= approx_img)

# prediction_img = disp_pyr_vertical(prediction_pyr)
prediction_img = disp_pyr_horizontal(prediction_pyr)
plt.imshow(prediction_img,cmap='gray')
plt.show()
cv.imwrite(filename='../result_img/prediction_pyramid.png', img= prediction_img)



# print(dst.shape)
# cv.imshow(winname='dst', mat=dst)
# cv.waitKey(0)

