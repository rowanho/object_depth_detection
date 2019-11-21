import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from feature_points import detect_matches
from object_detection import preprocess
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

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    print(alpha, beta)
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

def reduce_brightness():
    left_path = os.path.join(
        full_path_directory_left,
        '1506942502.475483_L.png')
    img = cv2.imread(left_path, cv2.IMREAD_COLOR)
    #processed_img = np.power(img, 1.2)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    b, g, r = cv2.split(img)
    red = clahe.apply(r)
    green = clahe.apply(g)
    blue = clahe.apply(b)
    processed_img = cv2.merge((blue, green, red))

    processed_img = processed_img.astype(np.uint8)
    cv2.imwrite('too_much_brightness.png', img)
    cv2.imwrite('reduce_brightness.png', processed_img)
    plt.show()
    


reduce_brightness()    
#plot_feature_points()
