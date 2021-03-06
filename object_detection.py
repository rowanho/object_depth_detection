import math

import cv2
import numpy as np

# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in


def drawPred(image, class_name, left, top, right, bottom, colour, depth):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s : %.2fm' % (class_name, depth)

    # Display the label at the top of the bounding box
    labelSize, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    
    cv2.rectangle(image,
                 (left, bottom + round(0.5 *labelSize[1])),
                 (left + round(1.0 * labelSize[0]), 
                  bottom +baseline - round(1.3 *labelSize[1])),
                  (255,255,255),
                  cv2.FILLED)
    cv2.putText(image, label, (left, bottom),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                

# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        threshold_confidence,
        threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)

# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

config_file = 'yolov3.cfg'
weights_file = 'yolov3.weights'

# Relevant YOLO classes

classes = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck'
    ]

name_map = {
    'person':'pedestrian',
    'bicycle': 'vehicle',
    'car': 'vehicle',
    'motorbike':'vehicle',
    'bus': 'vehicle',
    'truck': 'vehicle'
}

# load configuration and weight files for the model and load the network
# using them

net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
output_layer_names = getOutputsNames(net)

# defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib
# available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should
# fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Preprocesses the image for better obejct detection
# Power transform + histogram equalization
def preprocess(img):
    img = np.power(img, 0.95).astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    b, g, r = cv2.split(img)
    red = clahe.apply(r)
    green = clahe.apply(g)
    blue = clahe.apply(b)
    processed_img = cv2.merge((blue, green, red))
    return processed_img

# Kmeans estimation of foreground depth

def get_kmeans(box):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS    
    box = box.astype(np.float32)
    box = np.reshape(box, (-1, 1))
    
    # If there are too few elements
    if box.size < 2:
        return np.median(box)
    compactness,labels,centers = cv2.kmeans(box,2,None,criteria,10,flags)

    c0 = box[labels==0]
    c1 = box[labels==1]
    # Take the lower center
    m0 = np.mean(c0)
    m1 = np.mean(c1)
    return min(m0, m1)
    
# Estimates the depth of the object inside the box
# Use different methods based on whether we are using sparse disparity
def depth_estimate(box):
    res = 0
    non_zeros = box[np.nonzero(box)]
    if non_zeros.shape[0] == 0:
        res = 0
    else:
        # Apply KMeans
        res = get_kmeans(non_zeros)
    return res
            
# Applies yolo object detection, and draws labelled bounding boxes    
def apply_yolo(frame, depth_points, crop_y, crop_x, is_sparse, use_fg_mask):
    # Crop the frame to run the detection on
    cropped_frame = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    cropped_frame = preprocess(cropped_frame)
    # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1,
    # image resized)
    tensor = cv2.dnn.blobFromImage(
        cropped_frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(tensor)
    # runs forward inference to get output of the final output layers
    results = net.forward(output_layer_names)

    # remove the bounding boxes with low confidence
    confThreshold = 0.01
    classIDs, confidences, boxes = postprocess(
        cropped_frame, results, confThreshold, nmsThreshold)

    closest = 1000
    # draw resulting detections on image
    for detected_object in range(0, len(boxes)):
        if classIDs[detected_object] >= len(classes):
            continue
        box = boxes[detected_object]
        left = box[0] + crop_x[0]
        top = box[1] + crop_y[0]
        width = box[2]
        height = box[3]
        
        box_depth = depth_points[top: top + height, left:left + width]
        if box_depth.shape[0] == 0 or box_depth.shape[1] == 0:
            continue
        depth = depth_estimate(box_depth)
        # An esitmated depth of 0 usually is due to noise in the depth map
        if depth == 0:
            continue
        if depth < closest:
            closest = depth
        class_name =  classes[classIDs[detected_object]]
        
        if class_name not in name_map:
            continue
        drawPred(frame, name_map[class_name],
                 left, top, left + width, top + height, (255, 178, 50), depth)
    if closest == 1000:
        print(': nearest detected scene object (na m)')
    else:
        print(': nearest detected scene object (%2.1fm)' % closest)
        
