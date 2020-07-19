import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def meanFilter(src,kernel_size):
    filtered_img = src

    height, width = src.shape
    indexer = kernel_size // 2

    for i in range(indexer, height-indexer):
        for j in range(indexer, width-indexer):
            block = src[i-indexer: i+indexer+1, j-indexer : j+indexer+1]
            filtered_pixel_value = np.mean(block, dtype=np.float32)
            filtered_img[i][j] = int(filtered_pixel_value)
    return filtered_img


def box_filter_pixel(x, y, img, filter_size):
    normalization_coefficient = 1 / (filter_size * filter_size)
    kernel = (normalization_coefficient) * np.ones(shape=(filter_size, filter_size), dtype=np.uint16)

    filter_value = 0

    # filter_value += (img[x + i, y + j] * kernel[i,j])
    for i in range(filter_size):
        for j in range(filter_size):
            filter_value += (img[x + i, y + j] * kernel[i,j])

    return filter_value

def box_filter(img,filter_size):
    # normalization_coefficient = 1/(filter_size * filter_size)
    #
    # filter = (normalization_coefficient) * np.ones(shape= (filter_size,filter_size), dtype=float)

    height, width = img.shape
    # size = height * width
    zero_pad_row_count = (height - 1) // 2
    zero_pad_col_count = (width - 1) // 2


    zero_pad_up_count = zero_pad_row_count // 2
    # zero_pad_down_count = zero_pad_row_count // 2 + 1
    zero_pad_left_count = zero_pad_col_count // 2
    # zero_pad_right_count = zero_pad_col_count // 2 + 1


    padded_img = np.arange((height + zero_pad_row_count) * (width + zero_pad_col_count)).reshape(height + zero_pad_row_count, width + zero_pad_col_count)

    # np.full(filtered_img, fill_value=None)
    # temp = np.arange(size + zero_pad_row_count * width + zero_pad_col_count * height).reshape(height + zero_pad_row_count, width + zero_pad_col_count)
    # np.full(temp,fill_value=None)
    # filtered_img = np.full_like(shape=(zero_pad_up_count + height + zero_pad_down_count, zero_pad_left_count + width + zero_pad_right_count), fill_value= None)

    filtered_img = np.zeros_like(padded_img, dtype=float)
    padded_img[:, :] = 0

    padded_img[zero_pad_up_count : zero_pad_up_count + height, zero_pad_left_count : zero_pad_left_count + width] = img

    padded_height = padded_img.shape[0]
    padded_width = padded_img.shape[1]

    for x in range(padded_height):
        for y in range(padded_width):
            if x <= padded_height - filter_size and y <= padded_width - filter_size:
                filtered_img[x + 1, y + 1] = box_filter_pixel(x, y, padded_img, filter_size)


    return  filtered_img[zero_pad_up_count : zero_pad_up_count + height, zero_pad_left_count : zero_pad_left_count + width]



LOOP_COUNT = 200
img = cv.imread('../img/Lena.bmp', 0)

for i in range(LOOP_COUNT):
    img = meanFilter(src=img, kernel_size=3)

cv.imwrite('../result_img/filtered_img after 200 times.bmp',img)
cv.imshow('filtered_img after 200 times',img)
cv.waitKey(0)