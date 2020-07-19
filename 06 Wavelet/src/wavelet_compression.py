import pywt
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def coeff_quantize(c):
    height, width = c.shape
    c_prim = np.zeros_like(c)
    for u in range(height):
        for v in range(width):
            c_prim[u, v] = STEP_SIZE * np.sign(c[u, v]) * math.floor(c[u, v] / STEP_SIZE)

    return c_prim

def psnr(img, ref):

    _img = np.float64(np.copy(img))
    _ref = np.float64(np.copy(ref))

    mse = np.mean((_img - _ref) ** 2)

    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

STEP_SIZE = 2

src = cv.imread('../img/Lena.bmp', 0)
shape = src.shape

c = pywt.wavedec2(data=src, wavelet='haar', level=1)
# print(type(c)) # <class 'list'>
# print(c[0].shape) # (256, 256)
# print(c[1][0].shape) # (256, 256)
# print(c[1][1].shape) # (256, 256)
# print(c[1][2].shape) # (256, 256)

cA = coeff_quantize(c[0])
cH = coeff_quantize(c[1][0])
cV = coeff_quantize(c[1][1])
cD = coeff_quantize(c[1][2])

c_prim = [cA, (cH,cV,cD)]


# arr = pywt.coeffs_to_array(c)

recons = pywt.waverec2(coeffs=c_prim,wavelet='haar')

# print(type(recons)) # <class 'numpy.ndarray'>
# print(recons.shape) # (512, 512)

fig, axes = plt.subplots(1, 2)

axes[0].set_axis_off()
axes[0].imshow(src, cmap=plt.cm.gray)
axes[0].set_title('original')

axes[1].set_axis_off()
axes[1].imshow(recons, cmap=plt.cm.gray)
axes[1].set_title('reconstructed by wavelet')

fig_name = '../result_img/wavelet_compression.png'
fig.savefig(fig_name)

plt.subplot(121), plt.imshow(X=src, cmap='gray')
plt.title(label='original'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(X=recons, cmap='gray')
plt.title(label='reconstructed by wavelet'), plt.xticks([]), plt.yticks([])

plt.plot()
plt.show()
#
print('PSNR= ', psnr(src, recons))
#
