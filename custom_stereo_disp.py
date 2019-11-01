import cv2
import numpy as np

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)


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
    

# Returns 2d points and 3d depth with format [x(2d),y(2d),z(3d)]
# imgL - The left image
# imgR - The right image
def get_depth_points(imgL, imgR):
    # remember to convert to grayscale (as the disparity matching works on grayscale)
    # N.B. need to do for both as both are 3-channel images

    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    # perform preprocessing - raise to the power, as this subjectively appears
    # to improve subsequent disparity calculation

    grayL = np.power(grayL, 0.75).astype('uint8')
    grayR = np.power(grayR, 0.75).astype('uint8')

    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    disparity = stereoProcessor.compute(grayL,grayR)

    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5 # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)
        
    points = project_disparity_to_2d_with_depth(disparity_scaled, max_disparity)
    
    return points
