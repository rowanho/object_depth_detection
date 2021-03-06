import cv2
import numpy as np

orb = cv2.ORB_create(nfeatures=10000)
orb.setFastThreshold(0)

MIN_MATCH_COUNT = 10
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2


search_params = dict(checks=50)   # or pass empty dictionary
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Takes in grayscale images imgL and imgR
# Returns keypoints and the 'good' matches between them, using ORB feature
# point detection
def detect_matches(imgL, imgR, plot_info=False):
    
    # Detect orb feature points
    keypointsL, descriptors1 = orb.detectAndCompute(imgL, None)
    keypointsR, descriptors2 = orb.detectAndCompute(imgR, None)

    matches = []
    if (len(descriptors1) > 0):
        # get best matches (and second best matches)
        # using a k Nearst Neighboour (kNN) radial matcher with k=2
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    # Need to isolate only good matches, so create a mask

    # perform a first match to second match ratio test as original SIFT paper (known as Lowe's ration)
    # using the matching distances of the first and second matches

    good_matches = []
    try:
        for (m, n) in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    except ValueError:
        print("caught error - no matches from current frame")

    if len(good_matches) > MIN_MATCH_COUNT:

        # construct two sets of points - source (the selected object/region
        # points), destination (the current frame points)

        source_pts = np.float32(
            [keypointsL[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_pts = np.float32(
            [keypointsR[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # compute the homography (matrix transform) from one set to the other
        # using RANSAC

        H, mask = cv2.findHomography(
            source_pts, destination_pts, cv2.RANSAC, 5.0)

        matches_mask = mask.ravel().tolist()

        h, w = imgL.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)

        img2 = cv2.polylines(imgR, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print(
            "Not enough matches are found - %d/%d" %
            (len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None

    if plot_info:
        return keypointsL, keypointsR, good_matches, matches_mask
    else:
        return keypointsL, keypointsR, good_matches


# Computes the disparity map
def get_sparse_disp(imgL, imgR):
    keypointsL, keypointsR, good_matches = detect_matches(imgL, imgR)

    disp_img = np.zeros(imgL.shape, dtype=np.int16)
    for m in good_matches:
        xl = int(keypointsL[m.queryIdx].pt[0])
        yl = int(keypointsL[m.queryIdx].pt[1])
        xr = int(keypointsR[m.trainIdx].pt[0])
        # Disparity is the distance between the corresponding keypoints in the left and right images
        disp_img[yl, xl] = abs(xl - xr)
    return disp_img
