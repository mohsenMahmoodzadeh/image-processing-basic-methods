import cv2 as cv
import numpy as np


def affine_transform(v,w):
    src_coord = np.array(v,w,1)
    transformed_coord = np.empty([1,3],dtype=int)
    #transformed_coord = np.dot(src_coord , affine_matrix)


    #theta = degree * np.pi/180
    #x = int(v * np.cos(theta) - w * np.sin(theta))
    #y = int(v * np.sin(theta) + w * np.cos(theta))

    #return x,y

# def warp(src,M,size):
#     dst = np.zeros(size,dtype=np.uint8)
#     h , w = src.shape
#     print(dst.shape)
#     for u in range(size[0]):
#         for v in range(size[1]):
#             # dst[i, j] = int(src[M[0, 0] * i + M[0, 1] * j + M[0, 2], M[1, 0] * i + M[1, 1] * j + M[1, 2]])
#             x = int(M[0, 0] * (u) + M[0, 1] * (v) + M[0, 2])
#             y = int(M[1, 0] * (u) + M[1, 1] * (v) + M[1, 2])
#             if 0 < x < w  and 0 < y < h:سد
#                 # temp = src[v, w]
#                 dst[v, u] = src[x, y]
#                 # np.put(dst,[[i,j]],temp)
#     return dst

def getRotationMtrix2D(center_coordX,center_coordY,angle = 0.0,scale = 1.0):
    '''transform degree to radian'''
    theta = angle * np.pi/180

    alpha = scale * np.cos(theta)
    beta = scale * np.sin(theta)

    rotationMartix2D = np.array([[alpha, beta, (1-alpha)*center_coordX - beta * center_coordY],
                                 [-beta, alpha, beta * center_coordX + (1-alpha) * center_coordY]])

    return rotationMartix2D

def bilinear_interpolation(img, target_pixel):

    '''coefficients of interpolation'''
    a = b = c = d = None

    '''get coordination of pixel which should be interpolated'''
    x = target_pixel[0]
    y = target_pixel[1]

    '''get coordination and intensity of adjacent pixels '''
    top_left = img[x - 1][y - 1]
    top_left_x = x - 1
    top_left_y = y - 1

    top_right = img[x - 1][y + 1]
    top_right_x = x - 1
    top_right_y = y + 1

    down_right = img[x + 1][y + 1]
    down_right_x = x + 1
    down_right_y = y + 1

    down_left = img[x + 1][y - 1]
    down_left_x = x + 1
    down_left_y = y - 1

    '''matrix A, matrix of adjacent pixels coordination'''
    A = np.array([[top_left_x, top_left_y, top_left_x * top_left_y, 1]
            ,[top_right_x, top_right_y, top_right_x * top_right_y, 1]
            ,[down_right_x, down_right_y, down_right_x * down_right_y, 1]
            ,[down_left_x, down_left_y, down_left_x * down_left_y, 1]])

    '''matrix B, matrix of adjacent pixels intensity'''
    B = np.array([top_left, top_right, down_right, down_left])

    '''solve the linear equation to get the coefficients'''
    X = np.linalg.solve(A,B)

    a, b, c, d = X[0], X[1], X[2], X[3]

    '''return calculated intensity for target pixel'''
    return a * x + b * y + c * x * y + d

def warpAffine(src, M, size):

    '''create a matrix dst for destination matrix'''
    dst = np.zeros(size, dtype=np.uint8)
    '''getting the dimensions of source and destination images'''
    height, width = src.shape
    new_height, new_width = size[0], size[1]

    '''loop over source image to calculate the affine transform'''
    for x in range(height):
        for y in range(width):
            u = int(M[0, 0] * x + M[0, 1] * y + M[0, 2])
            v = int(M[1, 0] * x + M[1, 1] * y + M[1, 2])
            '''if destination coordinate doesn't exceed the dimension of destination image'''
            if 0 <= u < new_height and 0 <= v < new_width:
                dst[u, v] = src[x, y]

    '''loop over destination image to interpolate(bilinear interpolation here)'''
    final_image = np.zeros(size, dtype=np.uint8)
    for i in range(new_height-1):
        for j in range(new_width-1):
            '''ignore the frame of image(four tapes around the image) for interpolation'''
            if i != 0 or i != new_height - 2 or j != 0 or j != new_width - 2:
                final_image[i, j] = bilinear_interpolation(dst, (i, j))
                # if dst[i,j] == 0:
                #     final_image[i,j] = bilinear_interpolation(dst,(i,j))
    #return dst
    return final_image


if __name__ == "__main__":

    '''reading source image'''
    src_path = r'.\img\elaine.bmp'
    src = cv.imread(src_path, 0)

    height, width = src.shape

    '''finding center of image'''
    centerX, centerY = (width/2, height/2)

    '''calculate an affine matrix for 2D rotation'''
    rotation_angle = 80
    M = getRotationMtrix2D(centerX, centerY, rotation_angle)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    '''calculate the new width and new height to account for rotation'''
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # M[0,2] += centerX - (new_width/2)
    # M[1,2] += centerY - (new_height/2)

    M[0, 2] += (new_width/2) - centerX
    M[1, 2] += (new_height/2) - centerY

    #dst = cv.warpAffine(src,M,(new_width,new_height))

    '''calculate affine transform for source image'''
    dst = warpAffine(src, M, (new_height, new_width))

    cv.imwrite(r'.\result_images\rotated80.bmp',dst)

    '''show the rotated image'''
    win_name = 'rotated80'
    cv.imshow(win_name,dst)
    cv.waitKey(0)
    cv.destroyAllWindows()