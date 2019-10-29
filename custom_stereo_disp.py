import cv2
import numpy as np

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

# Returns the stereo disparity of the two images
# imgL - The left image
# imgR - The right image
def get_stereo_disp(imgL, imgR, crop_disparity=False,):
    # remember to convert to grayscale (as the disparity matching works on grayscale)
    # N.B. need to do for both as both are 3-channel images

    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);
    # perform preprocessing - raise to the power, as this subjectively appears
    # to improve subsequent disparity calculation

    grayL = np.power(grayL, 0.75).astype('uint8');
    grayR = np.power(grayR, 0.75).astype('uint8');

    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    disparity = stereoProcessor.compute(grayL,grayR);

    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5; # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);
    
    # crop disparity to chop out left part where there are with no disparity
    # as this area is not seen by both cameras and also
    # chop out the bottom area (where we see the front of car bonnet)

    if (crop_disparity):
        width = np.size(disparity_scaled, 1);
        disparity_scaled = disparity_scaled[0:390,135:width];
    
    return disparity_scaled * (256. / max_disparity)).astype(np.uint8)
