import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def filter_pipeline(f,h):
    m,n = f.shape
    '''padding image'''
    p = 2 * m
    q = 2 * n
    '''zero padded image'''
    f_p = np.zeros(shape=(p, q), dtype=np.uint8)
    f_p[:m, :n] = f
    '''DFT'''
    F = np.fft.fft2(f_p)
    F_shift = np.fft.fftshift(F)
    # magnitude = 20 * np.log(np.abs(F_shift))
    '''operating real filter'''
    h_p = np.zeros(shape=(p, q), dtype=np.float)
    h_p[p // 2 - 1: p // 2 + 2, q // 2 - 1: q // 2 + 2] = h
    H = np.fft.fft2(h_p)
    H_shift = np.fft.fftshift(H)
    HF_p = np.multiply(F_shift, H_shift)
    '''IDFT'''
    HF_p_unshifted = np.fft.ifftshift(x=HF_p)
    hf_p = np.fft.ifft2(a=HF_p_unshifted)
    '''choosing real part of image'''
    g_p = hf_p.real
    '''filtered_image'''
    g = g_p[m:p, n:q]
    return g


filename = '../img/Barbara.bmp'
barbara = cv.imread(filename=filename, flags=0)


a_filter = 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) # smoothing filter
b_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) # edge detection
c_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # edge enhancement

A_filter = np.fft.fft2(a=a_filter)
A_filter_shift = np.fft.fftshift(A_filter)
A_filter_magnitude_log_shift = np.log(1 + np.abs(A_filter_shift))
A_filter_magnitude_log = np.log(1 + np.abs(A_filter))

B_filter = np.fft.fft2(a=b_filter)
B_filter_shift = np.fft.fftshift(B_filter)
B_filter_magnitude_log_shift = np.log(1 + np.abs(B_filter_shift))
B_filter_magnitude_log = np.log(1 + np.abs(B_filter))

C_filter = np.fft.fft2(a=c_filter)
C_filter_shift = np.fft.fftshift(c_filter)
C_filter_magnitude_log_shift = np.log(1 + np.abs(C_filter_shift))
C_filter_magnitude_log = np.log(1 + np.abs(C_filter))


plt.subplot(131), plt.imshow(X=c_filter, cmap='gray')
plt.title(label='filter (c)'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(X=C_filter_magnitude_log, cmap='gray')
plt.title(label='magnitude w.o shift'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(X=C_filter_magnitude_log_shift, cmap='gray')
plt.title(label='magnitude with shift'), plt.xticks([]), plt.yticks([])
plt.savefig('../result_img/c_filter_magnitudes.png')
plt.show()
a_filtered_barbara = filter_pipeline(f=barbara, h=a_filter)
b_filtered_barbara = filter_pipeline(f=barbara, h=b_filter)
c_filtered_barbara = filter_pipeline(f=barbara, h=c_filter)
cv.imwrite(filename='../result_img/a_filtered_barbara.bmp', img=a_filtered_barbara)
cv.imwrite(filename='../result_img/b_filtered_barbara.bmp', img=b_filtered_barbara)
cv.imwrite(filename='../result_img/c_filtered_barbara.bmp', img=c_filtered_barbara)
def immse(A, B):
    return (np.square(A - B)).mean()

print(immse(a_filter, A_filter_magnitude_log_shift))


Component1 = np.fft.fft( np.array([1, 2, 1]))
Component2 = np.fft.fft( np.array([[1], [2], [1]]))
# print(Component1.shape)
# print(Component2.shape)
# A_filter = np.fft.fft2(a_filter)


# con = np.convolve(a=Component2.reshape((3,)), v=Component1, mode='same')
# print(1/16 * Component1 * Component2)
# print()
# print(A_filter)

# print(A_filter)






