import numpy as np
import cv2

import matplotlib.pyplot as plt
feature_object = cv2.ORB_create(800)
MIN_MATCH_COUNT = 10
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
table_number = 6, # 12
key_size = 12,     # 20
multi_probe_level = 1) #2



search_params = dict(checks=50)   # or pass empty dictionary
matcher = cv2.FlannBasedMatcher(index_params,search_params)


def detect_matches(gray_frame1, gray_frame2):
    # if using ORB points use FLANN object that can handle binary descriptors
    # taken from: https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
    # N.B. "commented values are recommended as per the docs,
    # but it didn't provide required results in some cases"

    # get best matches (and second best matches)
    # using a k Nearst Neighboour (kNN) radial matcher with k=2
    keypoints1, descriptors1 = feature_object.detectAndCompute(gray_frame1,None)
    keypoints2, descriptors2 = feature_object.detectAndCompute(gray_frame2,None)

    matches = []
    if (len(descriptors1) > 0):
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    # Need to isolate only good matches, so create a mask

    # perform a first match to second match ratio test as original SIFT paper (known as Lowe's ration)
    # using the matching distances of the first and second matches

    good_matches = []
    try:
       for (m,n) in matches:
           if m.distance < 0.7*n.distance:
               good_matches.append(m)
    except ValueError:
        print("caught error - no matches from current frame")
    

    if len(good_matches)>MIN_MATCH_COUNT:

        # construct two sets of points - source (the selected object/region points), destination (the current frame points)

        source_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        destination_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        # compute the homography (matrix transform) from one set to the other using RANSAC

        H, mask = cv2.findHomography(source_pts, destination_pts, cv2.RANSAC, 5.0)

        matchesMask = mask.ravel().tolist()
        
        h,w = gray_frame1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)

        img2 = cv2.polylines(gray_frame2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    print(keypoints1)
    print(keypoints2)
    print(good_matches)
    return keypoints1, keypoints2, good_matches
    
    """    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(gray_frame1,keypoints1,gray_frame2,keypoints2,good_matches,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()    
    """


#returns the disparity map
def get_disp_map(img1, img2):
    keypoints1, keypoints2, good_matches  = detect_matches(img1, img2)
    
    disp_img = np.zeros(img1.shape)
    print(disp_img)
    print(img1)
    for m in good_matches:
        xl = int(keypoints1[m.queryIdx].pt[0])
        yl = int(keypoints1[m.queryIdx].pt[1])
        xr = int(keypoints2[m.trainIdx].pt[0])
        disp_img[yl, xl] = abs(xl - xr)
        
    
    print(disp_img)
    #return disp_img    
