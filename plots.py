import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from feature_points import detect_matches

master_path_to_dataset = "TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"
full_path_directory_left = os.path.join(
    master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(
    master_path_to_dataset, directory_to_cycle_right)


def plot_feature_points():
    left_path = os.path.join(
        full_path_directory_left,
        '1506942592.475323_L.png')
    right_path = os.path.join(
        full_path_directory_right,
        '1506942592.475323_R.png')

    imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    keypointsl, keypointsr, good_matches, matches_mask = detect_matches(
        imgL, imgR, plot_info=True)
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(
        imgL,
        keypointsl,
        imgR,
        keypointsr,
        good_matches,
        None,
        **draw_params)

    plt.imshow(img3, 'gray'), plt.savefig('orb_feature_plot.png')
    plt.show()


plot_feature_points()
