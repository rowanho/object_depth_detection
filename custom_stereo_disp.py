import cv2
import numpy as np
from surf import get_sparse_disp
max_disparity = 64
left_matcher = cv2.StereoSGBM_create(0, max_disparity, 21)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(80000)
wls_filter.setSigmaColor(1.2)


# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5

## Numpy vectorised function to calcuate depth
def depth(disp, f, B):
    if disp > 0:
        return (f * B) / disp
    else:
        return 0.0
        
        
## Project a given disparity image to have 3d depth points

def project_disparity_to_2d_with_depth(disparity, max_disparity):
    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    height, width = disparity.shape[:2]
    
    points = np.zeros((height,width))
    vec_depth = np.vectorize(depth)
    
    points = vec_depth(disparity, f, B)
    return points



def disp_with_wls_filtering(imgL, imgR):
    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredDisp = wls_filter.filter(displ, imgL, None, dispr)
    return filteredDisp                                    
    
def preprocess(img):
    img = np.power(img, 0.85).astype('uint8')
    img = cv2.equalizeHist(img)
    return img
    
# Returns 2d points and 3d depth with format [x(2d),y(2d),z(3d)]
# imgL - The left image
# imgR - The right image
def get_depth_points(imgL, imgR, is_sparse):
    
    # remember to convert to grayscale (as the disparity matching works on grayscale)
    # N.B. need to do for both as both are 3-channel images

    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

    # perform preprocessing - raise to the power, as this subjectively appears
    # to improve subsequent disparity calculation

    grayL = preprocess(grayL)
    grayR = preprocess(grayR)
    if is_sparse:
        disparity =  get_sparse_disp(grayL, grayR)        

    else:        
        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)
        dispNoiseFilter = 10 # increase for more agressive filtering
        disparity = disp_with_wls_filtering(grayL, grayR)
        # filter out noise and speckles (adjust parameters as needed)
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)
        



    
    # Apply wls filter to disparity
    
    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity 
    #disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    _, disparity_scaled = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
    print(np.max(disparity_scaled))
    if not is_sparse:
        disparity_scaled = (disparity/ 16).astype(np.uint8)
    else:
        disparity_scaled = disparity
    cv2.imshow('disparity', (disparity_scaled * (256 / max_disparity)).astype(np.uint8))
    points = project_disparity_to_2d_with_depth(disparity_scaled, max_disparity)
    return points
