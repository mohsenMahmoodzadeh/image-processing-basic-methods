import pywt
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

src = cv.imread('../img/Lena.bmp', 0)
shape = src.shape

max_lev = 3
label_levels = 3

wavelet_coeffs_level_one = None

fig, axes = plt.subplots(1, 4, figsize=[14, 8])
for level in range(0, max_lev + 1):
    if level == 0:

            axes[0].set_axis_off()
            axes[0].imshow(src, cmap=plt.cm.gray)
            axes[0].set_title('original')
            # axes[0].set_axis_off()
            continue

    # if level == 1:
    #     c = pywt.wavedec2(src, 'haar', level=level)
    #     wavelet_coeffs_level_one = c
    # else:
    c = pywt.wavedec2(src, 'haar', level=level)

    c[0] /= np.abs(c[0]).max()
    for detail_level in range(level):
        c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]

    arr, slices = pywt.coeffs_to_array(c)

    axes[level].imshow(arr, cmap=plt.cm.gray)
    axes[level].set_title('Coefficients\n({} level)'.format(level))
    axes[level].set_axis_off()
    fig_name = '../result_img/wavelet_level_{}.png'.format(level)
    extent = axes[level].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fig_name, bbox_inches=extent.expanded(1.1, 1.2))
    # plt.savefig(fig_name, bbox_inches="tight")



# fig, axes = plt.subplots(2, 4, figsize=[14, 8])
# for level in range(0, max_lev + 1):
#     if level == 0:
#
#         axes[0, 0].set_axis_off()
#         axes[1, 0].imshow(src, cmap=plt.cm.gray)
#         axes[1, 0].set_title('original')
#         axes[1, 0].set_axis_off()
#         continue
#
#
#     draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
#                      label_levels=label_levels)
#     axes[0, level].set_title('{} level\ndecomposition'.format(level))
#
#
#     # c = pywt.wavedec2(src, 'haar', mode='periodization', level=level)
#     c = pywt.wavedec2(src, 'haar', level=level)
#
#     c[0] /= np.abs(c[0]).max()
#     for detail_level in range(level):
#         c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
#
#     arr, slices = pywt.coeffs_to_array(c)
#     axes[1, level].imshow(arr, cmap=plt.cm.gray)
#     axes[1, level].set_title('Coefficients\n({} level)'.format(level))
#     axes[1, level].set_axis_off()
#
plt.tight_layout()
# plt.savefig('../result_img/wavelet_pyramid 2.png', bbox_inches="tight")
plt.show()





# coeffs = pywt.dwt2(data=src ,wavelet='haar')
# LL, (LH, HL, HH) = coeffs
#
# print(src.shape)
# print(LL.shape)
# plt.subplot(121), plt.imshow(X=src, cmap='gray')
# plt.title(label='original'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(122), plt.imshow(X=LL, cmap='gray')
# plt.title(label='LL1'), plt.xticks([]), plt.yticks([])
#

# plt.show()

