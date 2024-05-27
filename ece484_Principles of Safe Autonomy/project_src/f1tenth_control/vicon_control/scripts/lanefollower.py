#!/usr/bin/env python3

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64, Float64MultiArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

import time
import math
import numpy as np
import cv2
import rospy

from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from skimage import morphology


class LaneFollower():
    def __init__(self):
        self.bridge = CvBridge()
        # Subsriber and Publisher
        self.sub_img = rospy.Subscriber('/D435I/color/image_raw', Image, self.img_callback, queue_size=1)
        self.laser = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)
        self.pub_annotated_warp_img = rospy.Publisher("lane_detection/annotated_warp_img", Image, queue_size=1)
        self.pub_annotated_img = rospy.Publisher("lane_detection/annotated_img", Image, queue_size=1)
        self.pub_control = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"

        self.M = self.solve_perspective_transform()  # Perspective transform from image plane to ground (3, 3)

        self.L = 12.795276  # Wheel base length (inches)
        
        # Record current velocity and steering
        self.velocity = 0
        self.steering = 0

        # Record velocities and steerings for each run 
        self.steerings = []
        self.velocities = []

        # Variables for obstacle avoidance
        self.not_detect = 0
        self.try_detect = 0
        self.count_down = 0
        self.lidar = []  # Store x-y position of lidar points
        self.detect = False
        self.recovery_factor = 1  # Track the car turns left or right from the lane

    def img_callback(self, data):
        """
        Callback function for RGB image from camera
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  # Convert a ROS image message into an OpenCV image
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        thres_out = self.filter_img(raw_img)  # Filter
        warp_out, M, M_inv = self.warp_img(thres_out)  # Warp to bird-eye view
        ret = self.find_waypoints(warp_out)  # Get waypoints on warped image plane
        out_img = ret['out_img']
        warped_img_wps = ret['centers']  # (..., 2)
        lane_back = False  # Flag to indicate lane can be detected
        img_wps_vis = None  # Waypoints on the image plane
        self.try_detect += 1 
        if warped_img_wps.size != 0:
            img_wps_vis, ground_wps = self.transform_waypoints(warped_img_wps, M_inv)  # Transform waypoints to car body frame
            velocity, steering = self.output_control(ground_wps)
            self.velocity = velocity
            self.steering = steering
            lane_back = True 
        else:
            velocity = self.velocity
            steering = self.steering
            self.not_detect+=1

        human_ahead = self.is_human_ahead(raw_img)  # Human detector
       
        if(human_ahead | self.detect):
            velocity = 0
            steering = 0
            self.detect = True

        if(len(self.lidar)):
            x = self.lidar[0][0]
            y = self.lidar[0][1]
            velocity = 0.5
            steering = math.atan2(y, x) * self.recovery_factor
            self.count_down = 5
            self.detect = True
        elif(self.detect):
            if(self.count_down == 0):
                self.count_down = 15
                self.detect = False
            velocity = 0.5
            steering = 0
            self.count_down -= 1
        elif(self.count_down):
            self.count_down -= 1
            if(lane_back):
                self.count_down = 0
            else:
                velocity = 0.5
                steering = 5 * self.recovery_factor

        print(f'v: {velocity} m/s, steering: {steering * (180 / np.pi)} degrees')

        # Publish control commands
        self.drive_msg.header.stamp = rospy.get_rostime()
        self.drive_msg.drive.steering_angle = steering
        self.drive_msg.drive.speed = velocity
        self.pub_control.publish(self.drive_msg)

        '''
        # Plot purpose
        self.steerings.append(steering)
        self.velocities.append(velocity)
        steering_arr = np.array(self.steerings)
        velocity_arr = np.array(self.velocities)
        np.save('steerings.npy', steering_arr)
        np.save('velocities.npy', velocity_arr)
        '''

        # Visualize waypoints on the image plane
        if img_wps_vis is not None:
            for i in range(img_wps_vis.shape[1]):
                x = int(img_wps_vis[0][i])
                y = int(img_wps_vis[1][i])
                raw_img = cv2.circle(raw_img, (x, y), 3, (0, 255, 0), -1)
        
        out_annotated_img = self.bridge.cv2_to_imgmsg(raw_img, 'bgr8')
        self.pub_annotated_img.publish(out_annotated_img)

        if out_img is not None:
            out_annotated_warp_img = self.bridge.cv2_to_imgmsg(out_img, 'bgr8')
            self.pub_annotated_warp_img.publish(out_annotated_warp_img)

    def laser_callback(self, data):
        """
        Callback function for LiDAR readings
        """
        self.lidar = []
        ranges = data.ranges
        angle = data.angle_min 
        increment = data.angle_increment 

        collision_dist = 0.8
        ignore_dist = 1
        car_boundary = 0.25
        left = 0  # Leftmost point
        right = 0  # Rightmost point
        for distance in ranges:
            angle += increment
            x = distance * np.cos(angle) 
            y = distance * np.sin( angle )
            if(distance > ignore_dist or x < 0):
                # Filter points too far away or in the back
                continue 
            if((y > -0.3) and (y < 0.3) and (x < collision_dist)):
                # Points lying in the front of the car 
                if(right):
                    left = [x, y + car_boundary]
                else: 
                    right = [x, y - car_boundary]
        if left and right:
            if abs(right[1]) > abs(left[1]):
                self.lidar.append(left)
                self.recovery_factor = -1
            else:
                self.lidar.append(right)
                self.recovery_factor = 1
        return 0 
    
    def is_human_ahead(self, pic):
        """
        HOG Descriptor for human detection
        Return 0 if there is human in front, else return 1
        """
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        size_threash = 10000
        frame = pic
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes_o, weights = hog.detectMultiScale(frame, winStride=(8,8))
        for i in range(len(weights)):
            if(weights[i] > 1):
                if(boxes_o[i][2] * boxes_o[i][3] > size_threash):
                    return 1
        return 0

    def filter_img(self, img):
        """
        Filter the image using color and gradient threshold
        """
        grad_thresh_min = 25
        grad_thresh_max = 100
        color_thresh = (100, 255)

        sobel_output = self.gradient_thresh(img, thresh_min=grad_thresh_min, thresh_max=grad_thresh_max)
        color_output = self.color_thresh(img, thresh=color_thresh)

        binary_img = np.zeros_like(sobel_output)
        binary_img[(color_output == 1) ] = 1
        
        binary_img = morphology.remove_small_objects(binary_img.astype('bool'),min_size=50,connectivity=2)  # Remove noise from binary image

        return binary_img        

    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x direction
        """
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_image, (5,5), 0)
        gX = cv2.Sobel(blur_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        gY = cv2.Sobel(blur_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

        combined = cv2.addWeighted(gX, 1, gY, 0, 0)  
        combined = cv2.convertScaleAbs(combined)

        # Apply threshold
        binary_output = np.zeros((img.shape[0], img.shape[1]))
        mask_1 = combined >= thresh_min 
        mask_2 = combined <= thresh_max
        mask = mask_1 & mask_2
        binary_output[mask] = 1
        binary_output = cv2.convertScaleAbs(binary_output)  # Convert to unint8

        return binary_output

    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and apply color threshold
        """
        # Threshold for yellow lane
        yellow_thres_low = np.array([10, 15, 70], dtype = np.uint8)
        yellow_thres_up = np.array([80, 249, 255], dtype = np.uint8)

        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        yellow_mask = cv2.inRange(img_hls, yellow_thres_low, yellow_thres_up)
        mask = yellow_mask

        filtered_hls = cv2.bitwise_and(img_hls, img_hls, mask=mask)
        filtered_bgr = cv2.cvtColor(filtered_hls, cv2.COLOR_HLS2BGR)
        filtered_gray = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY)
        filtered_gray[filtered_gray > 0] = 1 
    
        return filtered_gray

    def warp_img(self, img):
        """
        Get bird's eye view of the filtered image
        """
        # image size: (480, 640)
        src_pts = np.float32([[42, 350], [600, 350], [42, 480], [600, 480]]) 
        dst_pts = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        Minv = np.linalg.inv(M)

        warped_img_size = (img.shape[1], img.shape[0])
        warped_img = cv2.warpPerspective(img.astype("uint8"), M, warped_img_size)

        return warped_img, M, Minv
    
    def find_waypoints(self, binary_warped):
        """
        Find waypoints in warped image coordinate based on the bird-eye view
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)  # Take a histogram of the bottom half of the image
        out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) * 255).astype('uint8')  # Create an output image to draw on and visualize the result
        x_base = np.argmax(histogram[50: -50]) + 100  # Find the peak of the histogram
        nwindows = 6  # Choose the number of sliding windows
        window_height = np.int_(binary_warped.shape[0] / nwindows)  # Set height of windows

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        x_current = x_base  # Current positions to be updated for each window
        margin = 150  # Set the width of the windows +/- margin
        minpix = 40  # Set minimum number of pixels found to recenter window
        lane_inds = []  # Create empty lists to receive lane pixel indices

        y_0 = binary_warped.shape[0]  # Image height
        x_0 = binary_warped.shape[1]  # Image width
        array = np.argwhere(binary_warped != 0)  # All non-zero coordinates

        centers = []  # Empty list to store centers of windows

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y 
            LU_pixel = (x_current - margin, y_0 - (window + 1) * window_height) 
            RD_pixel = (x_current + margin, y_0 - (window) * window_height)

            # Draw the windows on the visualization image using cv2.rectangle()
            out_img = cv2.rectangle(out_img, LU_pixel, RD_pixel, (0,255,255), 5)

            # Identify the nonzero pixels in x and y within the window
            # Non-zero coordinates in the window
            x_min = LU_pixel[1]
            x_max = RD_pixel[1] 
            y_min = LU_pixel[0]
            y_max = RD_pixel[0]
            ranges = (array[:, 0] >= x_min) & (array[:, 0] <= x_max) & (array[:, 1] >= y_min) & (array[:, 1] <= y_max)
            coords = array[ranges]

            if coords[:, 1].size == 0 or coords[:, 0].size == 0:
                continue

            # Get the coordinate of the window center
            center_x = int(np.mean(coords[:, 1]))
            center_y = int(np.mean(coords[:, 0]))
            if (center_x >= 0 and center_x < x_0) and (center_y >= 0 and center_y < y_0):
                centers.append([center_x, center_y])
        
            # Draw the center for visualization
            out_img = cv2.circle(out_img, (center_x, center_y), 5, (0, 255, 0), -1)

            # If you found > minpix pixels, recenter next window on their mean position
            if (coords.shape[0] > minpix):
                x_current = int(np.mean(coords[:, 1]))
            
        # Return a dict of relevant variables
        ret = {}
        ret['out_img'] = out_img
        ret['centers'] = np.array(centers)

        return ret
    
    def solve_perspective_transform(self):
        """
        Solve perspective transformation from image plane to ground
        """
        # Manual tweaking for transformation
        hn = -0.3
        shrink = 1.4
        shorten = 1
        src_pts = np.float32([[180, 476], [565, 472], [248, 372], [477, 370]]) 
        dst_pts = np.float32([[15.25, 5 + hn], [15.25, -5 + hn], [30, 4 + hn], [30, -6 + hn]])
        dst_pts[:,0] /= shorten
        dst_pts[:,1] /= shrink
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        return M

    def transform_waypoints(self, warped_img_wps, M_inv):
        """
        Transform waypoints from bird-eye view to ground
        """
        # Transform to coordinates on the image plane
        num_wps = warped_img_wps.shape[0]
        ones_arr = np.ones((num_wps, 1))
        homo_warped_img_wps = np.hstack((warped_img_wps, ones_arr))
        img_wps = M_inv @ homo_warped_img_wps.T
        norm_values = img_wps[2:][0]
        img_wps_norm = img_wps / norm_values
        img_wps_vis = img_wps_norm[0:2, :]  # For visualization

        # Transform to coordinate in camera frame
        ground_wps = self.M @ img_wps_norm
        norm_values = ground_wps[2:][0]
        ground_wps_norm = ground_wps / norm_values
        ground_wps_norm = ground_wps_norm[0:2, :]  # (2, ...)

        return img_wps_vis, ground_wps_norm
    
    def output_control(self, ground_wps):
        """
        Pure pursuit controller
        """
        ####################### Longititudal Controller #######################
        target_velocity = 0.5  # Constant velocity
        #######################################################################

        ######################### Lateral Controller ##########################
        target_steering = 0
        # Add weights to the first and last waypoint and determine the lookahead point
        fake_target = None
        if (ground_wps.shape[1] > 1):
            fake_target = [(ground_wps[0][0] + ground_wps[0][0] + ground_wps[0][1] + 2 * ground_wps[0][-1])/ 5, (ground_wps[1][0] + ground_wps[1][0]  + ground_wps[1][1] + 2 * ground_wps[1][-1]) / 5]
        else:
            fake_target = [ground_wps[0][0], ground_wps[0][0]]
        ld = math.sqrt((fake_target[0]) ** 2 + (fake_target[1]) ** 2)

        alpha = math.atan2(fake_target[1], fake_target[0])  # Solve for alpha

        target_steering = math.atan2(2 * self.L * math.sin(alpha), ld) * 0.5
        #######################################################################

        return target_velocity, target_steering

        
if __name__ == '__main__':
    rospy.init_node('lanefollower_node', anonymous=True)
    rate = rospy.Rate(15)  # Hz
    lane_follower = LaneFollower()
    try: 
        while not rospy.core.is_shutdown():
            rate.sleep()
    except rospy.ROSInterruptException:
        pass



    



