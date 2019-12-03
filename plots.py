import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from feature_points import detect_matches

from object_detection import apply_yolo
from stereo import get_depth_points

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
        '1506942550.476061_L.png')
    img = cv2.imread(left_path, cv2.IMREAD_COLOR)
    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    lab[...,0] = clahe.apply(lab[...,0])
    processed_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    
    processed_img = processed_img.astype(np.uint8)
    plot_histogram(img, 'before_CLAHE_hist.png', 'Histogram before' )
    plot_histogram(processed_img,' after_CLAHE_hist.png', 'Histogram after')
    cv2.imwrite('before_CLAHE.png', img)
    cv2.imwrite('after_CLAHE.png', processed_img)
    plt.show()

def stats_for_specific_img():
    left_path = os.path.join(
        full_path_directory_left,
        '1506942487.479214_L.png')
    imgL = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right_path = os.path.join(
        full_path_directory_right,
        '1506942487.479214_R.png')
    imgR = cv2.imread(right_path, cv2.IMREAD_COLOR)
    depth_points = get_depth_points(imgL, imgR, False, False)

    apply_yolo(imgL, depth_points, (0, 390), 
              (0, np.size(imgL, 1)), False, False)
    cv2.imwrite('boxes.png', imgL)
    
def plot_distributions():
    data = pd.read_csv('dense_data.csv', names = ['Mean', 'Median', 'Mode', 'Kmeans Lower Cluster Mean'])
    data = data[data['Mean'] > 0.0]   
    data = data[data['25th Percentile'] > 0.0]   
    data = data[data['Histogram Peak'] > 0.0]   
    sns.kdeplot(data[data.columns[0]],bw=.05)
    sns.kdeplot(data[data.columns[1]],bw=.05)
    sns.kdeplot(data[data.columns[2]],bw=.05)
    sns.kdeplot(data[data.columns[3]],bw=.05)

    plt.xlabel('Depth Prediction (metres)')
    plt.ylabel('Relative Frequency')
    plt.savefig('distr.png')
    
def plot_distributions():
    dense_data = pd.read_csv('dense.csv', names = ['Dense Stereo (25th Percentile)'])
    sparse_data = pd.read_csv('sparse.csv', names = ['Sparse Stereo (25th Percentile)'])

    dense_data = dense_data[dense_data['Dense Stereo (25th Percentile)'] > 0.0]   
    sparse_data = sparse_data[sparse_data['Sparse Stereo (25th Percentile)'] > 0.0]   

    sns.kdeplot(dense_data[dense_data.columns[0]],bw=.05)
    sns.kdeplot(sparse_data[sparse_data.columns[0]],bw=.05)

    plt.xlabel('Depth Prediction (metres)')
    plt.ylabel('Relative Frequency')
    plt.savefig('distr_comp.png')
    
if __name__ == "__main__":
    #reduce_brightness()    
    #plot_feature_points()
    #stats_for_specific_img()
    plot_distributions()