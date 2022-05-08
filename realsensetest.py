## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
# import pyvirtualcam
import math
import random
from scipy.spatial.transform import Rotation
from math import floor, ceil
from pythonosc import udp_client
import serial

########################################
######### open viewer ##########
########################################

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile=pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 15 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
intrinsics = depth_profile.get_intrinsics()
rs_angle = 0
R_camera_to_world = Rotation.from_euler('ZYX', [0,0,rs_angle+90], degrees=True).inv()

pole_pos = np.array([0.68,6.99,-2.2])

# intrinsics.width = 
def to3d(x,y,depth):
    p_camera = np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [x,y], depth))
    p_world = R_camera_to_world.apply(p_camera)
    return p_world - pole_pos

def nothing(x):
    pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0,255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 73, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

font = cv2.FONT_HERSHEY_SIMPLEX
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
color_frame = aligned_frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data()) # color image

ser = serial.Serial("/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0", 9600, timeout=0.01)

# with pyvirtualcam.Camera(width=color_image.shape[1], height=color_image.shape[0], fps=20) as cam:
lastwinds = 0
client = udp_client.SimpleUDPClient('172.26.162.185', 10000)
gameover = False
oldnumba = 8

oldx = 0
oldy = 0
oldz = 0
alpha = 0.7
while True:
    l = ser.readline()
    d = str(l).split(',')
    if (len(d) > 1):
        d = d[1:-1]
        d = [x == '#' for x in d] + [True]
        v = d.index(True)
        print(f"{v}, {d}")
        if v == 0:
            gameover = True
        if v == 8:
            gameover = False
        smoothnumba = oldnumba * 0.75 + v * 0.25
        oldnumba = smoothnumba
        client.send_message("/gameover", 1 if gameover else 0)
        client.send_message("/breadbowl", smoothnumba)
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data()) # depth image
    color_image = np.asanyarray(color_frame.get_data()) # color image

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Convert to HSV colour space
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Read the trackbar values
    lh = cv2.getTrackbarPos("LH", "Tracking")
    ls = cv2.getTrackbarPos("LS", "Tracking")
    lv = cv2.getTrackbarPos("LV", "Tracking")
    uh = cv2.getTrackbarPos("UH", "Tracking")
    us = cv2.getTrackbarPos("US", "Tracking")
    uv = cv2.getTrackbarPos("UV", "Tracking")

    # Create arrays to hold the minimum and maximum HSV values
    hsvMin = np.array([lh, ls, lv])
    hsvMax = np.array([uh, us, uv])
    
    # Apply HSV thresholds 
    mask = cv2.inRange(hsv, hsvMin, hsvMax)

    # Uncomment the lines below to see the effect of erode and dilate
    # mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)

    # The output of the inRange() function is black and white
    # so we use it as a mask which we AND with the orignal image
    color_image = cv2.bitwise_and(color_image, color_image, mask=mask)

    # Invert the mask
    reversemask = 255-mask

    # Adjust detection parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 100;
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 5000
    
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
    
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.25
    
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.25

    # Detect blobs
    detector = cv2.SimpleBlobDetector_create(params)

    # Run blob detection
    keypoints = detector.detect(reversemask)

    # Get the number of blobs found
    blobCount = len(keypoints)

    # Write the number of blobs found
    text = "Count=" + str(blobCount) 
    cv2.putText(color_image, text, (5,25), font, 1, (0, 255, 0), 2)
    
    Width = color_image.shape[1]
    Height = color_image.shape[0]

    if blobCount > 0:
        # Write X position of first blob
        blob_x = keypoints[0].pt[0]
        blob_y = keypoints[0].pt[1]
        blob_size = keypoints[0].size

        s = blob_size / 2
        depth_mean = np.nanmean(depth_image[ceil(blob_y-s):ceil(blob_y+s), ceil(blob_x-s):ceil(blob_x+s)])

        xyz = to3d(blob_x, blob_y, depth_mean * depth_scale)

        text2 = "X=" + "{:.2f}".format(xyz[0] )
        cv2.putText(color_image, text2, (5,50), font, 1, (0, 255, 0), 2)

        # Write Y position of first blob
        text3 = "Y=" + "{:.2f}".format(xyz[1])
        cv2.putText(color_image, text3, (5,75), font, 1, (0, 255, 0), 2)        
        
        # Write Y position of first blob
        text3 = "Z=" + "{:.2f}".format(xyz[2])
        cv2.putText(color_image, text3, (5,100), font, 1, (0, 255, 0), 2)        


        # Draw circle to indicate the blob
        cv2.circle(color_image, (int(blob_x),int(blob_y)), int(blob_size / 2), (0, 255, 0), 2)
        cv2.circle(depth_colormap, (int(blob_x),int(blob_y)), int(blob_size / 2), (0, 255, 0), 2)

        #xyz
        oldx = xyz[0] * (1-alpha) + alpha * oldx
        oldy = xyz[1] * (1-alpha) + alpha * oldy
        oldz = xyz[2] * (1-alpha) + alpha * oldz
        winds = math.degrees(math.atan2(oldx,oldy))/360
        diff = winds-lastwinds
        if diff > 0.5:
            diff -= 0.5
        if diff < -0.5:
            diff += 0.5
        lastwinds += diff

        xs = oldx/3 * Height / 2
        ys = oldy/3 * Height / 2
        if math.isnan(xs) or math.isnan(ys):
            continue
        cv2.line(color_image, (int(xs + Width/2),int(ys + Height/2)), (int(Width/2),int(Height/2)), (255, 0, 0), 2)

        # Write Size of first blob
        text4 = "w=" + "{:.2f}".format(winds)
        cv2.putText(color_image, text4, (5,125), font, 1, (0, 255, 0), 2)    
        
        client.send_message("/x", oldx)
        client.send_message("/y", oldy)
        client.send_message("/z", oldz)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape
    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))
    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)

# Stop streaming
pipeline.stop()