import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


filename = '../img/Lena.bmp'
lena = cv.imread(filename=filename, flags=0)
Lena = np.fft.fft2(a=lena)
Lena_shift = np.fft.fftshift(Lena)
Lena_magnitude_log = np.log(1 + np.abs(Lena))
Lena_magnitude_log_shift = np.log(1 + np.abs(Lena_shift))
Lena_magnitude = np.abs(Lena)
Lena_magnitude_shift = np.abs(Lena_shift)
plt.imsave(fname='../result_img/lena.png',arr=lena, cmap='gray')
plt.imsave(fname='../result_img/Lena_magnitude_log.png', arr=Lena_magnitude_log, cmap='gray')
plt.imsave(fname='../result_img/Lena_magnitude_log_shift.png', arr=Lena_magnitude_log_shift, cmap='gray')
plt.imsave(fname='../result_img/Lena_magnitude.png', arr=Lena_magnitude, cmap='gray')
plt.imsave(fname='../result_img/Lena_magnitude_shift.png', arr=Lena_magnitude_shift, cmap='gray')

filename = '../img/F16.bmp'
f16 = cv.imread(filename=filename, flags=0)
F16 = np.fft.fft2(a=f16)
F16_shift = np.fft.fftshift(F16)
F16_magnitude_log = np.log(1 + np.abs(F16))
F16_magnitude_log_shift = np.log(1 + np.abs(F16_shift))
F16_magnitude = np.abs(F16)
F16_magnitude_shift = np.abs(F16_shift)
plt.imsave(fname='../result_img/f16.png',arr=f16, cmap='gray')
plt.imsave(fname='../result_img/F16_magnitude_log.png', arr=F16_magnitude_log, cmap='gray')
plt.imsave(fname='../result_img/F16_magnitude_log_shift.png', arr=F16_magnitude_log_shift, cmap='gray')
plt.imsave(fname='../result_img/F16_magnitude.png', arr=F16_magnitude, cmap='gray')
plt.imsave(fname='../result_img/F16_magnitude_shift.png', arr=F16_magnitude_shift, cmap='gray')

filename = '../img/Baboon.bmp'
baboon = cv.imread(filename=filename, flags=0)
Baboon = np.fft.fft2(a=baboon)
Baboon_shift = np.fft.fftshift(Baboon)
Baboon_magnitude_log = np.log(1 + np.abs(Baboon))
Baboon_magnitude_log_shift = np.log(1 + np.abs(Baboon_shift))
Baboon_magnitude = np.abs(Baboon)
Baboon_magnitude_shift = np.abs(Baboon_shift)
plt.imsave(fname='../result_img/baboon.png',arr=baboon, cmap='gray')
plt.imsave(fname='../result_img/Baboon_magnitude_log.png', arr=Baboon_magnitude_log, cmap='gray')
plt.imsave(fname='../result_img/Baboon_magnitude_log_shift.png', arr=Baboon_magnitude_log_shift, cmap='gray')
plt.imsave(fname='../result_img/Baboon_magnitude.png', arr=Baboon_magnitude, cmap='gray')
plt.imsave(fname='../result_img/Baboon_magnitude_shift.png', arr=Baboon_magnitude_shift, cmap='gray')