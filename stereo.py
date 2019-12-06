import cv2
import numpy as np

from feature_points import get_sparse_disp
max_disparity = 64
left_matcher = cv2.StereoSGBM_create(0, max_disparity, 9,P1 = 5, P2 = 5)# mode=cv2.StereoSGBM_MODE_HH)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.3)


# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5


# Create the background subtractor object
back_sub = cv2.createBackgroundSubtractorMOG2()

# Gets the foreground mask
def get_fg_mask(img):
     return back_sub.apply(img)


# Numpy vectorised function to calcuate depth
def depth(disp, f, B):
    if disp > 0:
        return (f * B) / disp
    else:
        return 0.0

# Project a given disparity image to have 2d depth points

def project_disparity_to_2d_with_depth(disparity, max_disparity):
    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    height, width = disparity.shape[:2]

    points = np.zeros((height, width))

    vec_depth = np.vectorize(depth)
    points = vec_depth(disparity, f, B)
    return points

# Computes the disparity using a wls filtering method
def disp_with_wls_filtering(imgL, imgR):
    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    # Apply wls filter to disparity
    filteredDisp = wls_filter.filter(displ, imgL, None, dispr)
    return filteredDisp

# Preprocessing for dense implementation
def preprocess_dense(img):
    img = np.power(img, 0.95).astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    
    return img
    
# Post processing for the background mask
def post_process_for_bg(mask):
    # Remove noisy pieces of foreground
    # Find connected components 
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    # Remove background component
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles
    min_num = 15

    cleaned_img = np.zeros((output.shape), dtype=np.uint8)
    
    # Only keep components larger than min_num
    for i in range(0, nb_components):
        if sizes[i] >= min_num:
            cleaned_img[output == i + 1] = 255
            
    kernel = np.ones((7,7),np.uint8)        
    cleaned_img = cv2.morphologyEx(cleaned_img, cv2.MORPH_CLOSE, kernel)
    return cleaned_img
    
# Preprocessing for sparse implementation    
def preprocess_sparse(img):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img
    

# Returns 2d points and 3d depth with format [x(2d),y(2d),z(3d)]

def get_depth_points(imgL, imgR, is_sparse, use_fg_mask):

    # convert to grayscale

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Different methods based on sparse and dense implementations
    if is_sparse:
        grayL = preprocess_sparse(grayL)
        grayR = preprocess_sparse(grayR)
        disparity = get_sparse_disp(grayL, grayR)

    else:
        grayL = preprocess_dense(grayL)
        grayR = preprocess_dense(grayR)
        # Disparity using left and right matching + wls filter
        disparity = disp_with_wls_filtering(grayL, grayR)
        dispNoiseFilter = 10  # increase for more agressive filtering
        # filter out noise and speckles (adjust parameters as needed)
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)
        
        if use_fg_mask:
            fg_mask = get_fg_mask(imgL)
            fg_mask = post_process_for_bg(fg_mask)
            # Only keep foreground values
            disparity = (fg_mask/255).astype(np.uint8) * disparity 
            
    _, disparity_scaled = cv2.threshold(
        disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    
    if  is_sparse:
        disparity_scaled = disparity
    else:
        disparity_scaled = (disparity / 16).astype(np.uint8)
    
    cv2.imshow('Dense disparity', (disparity_scaled *
                                (256 / max_disparity)).astype(np.uint8))                     
    points = project_disparity_to_2d_with_depth(
        disparity_scaled, max_disparity)
    return points
