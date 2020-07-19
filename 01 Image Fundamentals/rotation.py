import cv2 as cv
import numpy as np

path = r'.\img\Elaine.bmp'
window_name = 'result'

elaine = cv.imread(path)


def rotate_coordinate(v,w,theta):
    x = int(v * np.cos(theta) - w * np.sin(theta))
    y = int(w * np.sin(theta) + v * np.cos(theta))

    return x,y


def rotate_image(image,angle):

    #rotation_matrix = np.array([[np.cos(angle), np.sin(angle), 0], [-1*np.sin(angle),np.cos(angle),0], [0,0,1]])

    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-1 * np.sin(angle), np.cos(angle)]])

    print(rotation_matrix.shape)

    width = np.size(image,0)
    length = np.size(image,1)


    #temp = np.ones((width + 1, length))

    #temp[:,:,-1] = image

    print(width)
    print(length)

    temp = np.empty_like(image)

    # cnt = 0

    for i in range(length):
        for j in range(width):
            x,y = rotate_coordinate(i,j,angle)
            #print(temp[x,y])
            #print(image[i,j])
            # cnt = cnt + 1
            if(x > 0 and y > 0 and x < length and y < width ):
                temp[x, y] = image[i, j]

            #temp[i,j] = np.dot([i,j],rotation_matrix)
    # print(cnt)
    return temp

rot = rotate_image(elaine, np.pi/6)

new_path = r'.\img'
cv.imwrite('rotated.bmp',rot)

cv.imshow(window_name,rot)
cv.waitKey(0)