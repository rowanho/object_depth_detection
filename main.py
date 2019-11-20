import cv2
import os
import numpy as np
from object_detection import yolo_net
from stereo import get_depth_points
# set to dataset path
master_path_to_dataset = "TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

skip_forward_file_pattern = ""  # set to timestamp to skip forward to

pause_playback = False  # pause until key press after each image


# resolve full directory location of data set for left / right images

full_path_directory_left = os.path.join(
    master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(
    master_path_to_dataset, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left))

is_sparse = True
# Loop through files
for filename_left in left_file_list:

    if ((len(skip_forward_file_pattern) > 0) and not(
            skip_forward_file_pattern in filename_left)):
        continue
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = ""

    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(
        full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(
        full_path_directory_right, filename_right)

    if ('.png' in filename_left) and (
            os.path.isfile(full_path_filename_right)):
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        cv2.imshow('left image', imgL)
        depth_points = get_depth_points(imgL, imgR, is_sparse)

        yolo_net(imgL, depth_points, is_sparse,
                 (0, 390), (0, np.size(imgL, 1)))
        cv2.imshow('Image with detection', imgL)

        print("-- files loaded successfully")
        print()
        # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF
        if (key == ord('x')):       # exit
            break  # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled)
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback)
    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()

# close all windows

cv2.destroyAllWindows()
