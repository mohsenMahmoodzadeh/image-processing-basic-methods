import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from skimage.metrics import structural_similarity as compare_ssim
import sys
import statistics

def mse(A, B, ax=None):
    '''
    ax = 0: the average is performed along the row, for each column, returning an array
    ax = 1: the average is performed along the column, for each row, returning an array
    ax = None: the average is performed element-wise along the array, returning a scalar value
    '''
    return ((A - B)**2).mean(axis=ax)

def ssim(A, B):
    (score, diff) = compare_ssim(A, B, full=True)
    diff = (diff * 255).astype("uint8")
    return diff, score

def get_transform_matrix(image1, image2):

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2_gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(matches) * 0.15)
    matches = matches[:num_good_matches]
    
    num_matches = len(matches)

    feature_matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    return homography, num_matches, feature_matched_image

def match(query, reference, image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    homography, num_matches, feature_matched_image = get_transform_matrix(query, reference)
    matched_image = cv2.warpPerspective(image_gray, homography, (reference.shape[1], reference.shape[0]))
    return matched_image, feature_matched_image, num_matches

def stddev(x):
    mean = statistics.mean(tuple(x))
    return math.sqrt(sum([(val - mean)**2 for val in x])/(len(x) - 1))


reference_color = cv2.imread('./Images/7/Reference.bmp')
reference_gray = cv2.cvtColor(reference_color, cv2.COLOR_BGR2GRAY)

original_color = cv2.imread('./Images/7/Original.bmp')
original_gray = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)

attack1_1_color = cv2.imread('./Images/7/Attack 1/1.bmp')
attack1_1_gray = cv2.cvtColor(attack1_1_color, cv2.COLOR_BGR2GRAY)

attack1_2_color = cv2.imread('./Images/7/Attack 1/2.bmp')
attack1_2_gray = cv2.cvtColor(attack1_2_color, cv2.COLOR_BGR2GRAY)

attack1_3_color = cv2.imread('./Images/7/Attack 1/3.bmp')
attack1_3_gray = cv2.cvtColor(attack1_3_color, cv2.COLOR_BGR2GRAY)

attack1_4_color = cv2.imread('./Images/7/Attack 1/4.bmp')
attack1_4_gray = cv2.cvtColor(attack1_4_color, cv2.COLOR_BGR2GRAY)

attack2_1_color = cv2.imread('./Images/7/Attack 2/1.bmp')
attack2_1_gray = cv2.cvtColor(attack2_1_color, cv2.COLOR_BGR2GRAY)

attack2_2_color = cv2.imread('./Images/7/Attack 2/2.bmp')
attack2_2_gray = cv2.cvtColor(attack2_2_color, cv2.COLOR_BGR2GRAY)

attack2_3_color = cv2.imread('./Images/7/Attack 2/3.bmp')
attack2_3_gray = cv2.cvtColor(attack2_3_color, cv2.COLOR_BGR2GRAY)

attack2_4_color = cv2.imread('./Images/7/Attack 2/4.bmp')
attack2_4_gray = cv2.cvtColor(attack2_4_color, cv2.COLOR_BGR2GRAY)


matched_attack1_1, feature_matched_attack1_1, attack1_1_num_matches = match(attack1_1_color, reference_color, attack2_1_color)
matched_attack1_2, feature_matched_attack1_2, attack1_2_num_matches = match(attack1_2_color, reference_color, attack2_2_color)
matched_attack1_3, feature_matched_attack1_3, attack1_3_num_matches = match(attack1_3_color, reference_color, attack2_3_color)
matched_attack1_4, feature_matched_attack1_4, attack1_4_num_matches = match(attack1_4_color, reference_color, attack2_4_color)

mse_matched_attack1_1 = mse(original_gray, matched_attack1_1)
mse_matched_attack1_2 = mse(original_gray, matched_attack1_2)
mse_matched_attack1_3 = mse(original_gray, matched_attack1_3)
mse_matched_attack1_4 = mse(original_gray, matched_attack1_4)

ssim_matched_attack1_1 = ssim(original_gray, matched_attack1_1)
ssim_matched_attack1_2 = ssim(original_gray, matched_attack1_2)
ssim_matched_attack1_3 = ssim(original_gray, matched_attack1_3)
ssim_matched_attack1_4 = ssim(original_gray, matched_attack1_4)


mean_ssim = statistics.mean((ssim_matched_attack1_1, ssim_matched_attack1_2, ssim_matched_attack1_3, ssim_matched_attack1_4))
std_ssim = stddev([ssim_matched_attack1_1, ssim_matched_attack1_2, ssim_matched_attack1_3, ssim_matched_attack1_4])

mean_mse = statistics.mean((mse_matched_attack1_1, mse_matched_attack1_2, mse_matched_attack1_3, mse_matched_attack1_4))
std_mse = stddev([mse_matched_attack1_1, mse_matched_attack1_1, mse_matched_attack1_1, mse_matched_attack1_1])

mean_mp = statistics.mean((attack1_1_num_matches, attack1_2_num_matches, attack1_3_num_matches, attack1_4_num_matches))
std_mp = stddev([attack1_1_num_matches, attack1_2_num_matches, attack1_3_num_matches, attack1_4_num_matches])

data = {'SSIM': [ssim_matched_attack1_1, ssim_matched_attack1_2, ssim_matched_attack1_3, ssim_matched_attack1_4, mean_ssim, std_ssim],
        'MSE': [mse_matched_attack1_1, mse_matched_attack1_2, mse_matched_attack1_3, mse_matched_attack1_4, mean_mse, std_mse],
        'MP': [attack1_1_num_matches, attack1_2_num_matches, attack1_3_num_matches, attack1_4_num_matches, mean_mp, std_mp]
        }

index = ['Type1(Histeq)', 'Type2(Sharpen)', 'Type3(Gaussfilt)', 'Type4(Bilatfilt)', 'Mean', 'STD']
df = pd.DataFrame(data, index=index)

df.head()
