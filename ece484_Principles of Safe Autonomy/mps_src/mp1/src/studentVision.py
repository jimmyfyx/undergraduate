import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology



class lanenet_detector():
    def __init__(self):
        self.bridge = CvBridge()
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)

        # Uncomment this line for lane detection of videos in rosbag
        self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)

        # Uncomment this line for lane detection of videos in 0830
        # self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)

        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True

    def img_callback(self, data):
        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        # 1. Convert the image to gray scale
        # 2. Gaussian blur the image
        # 3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        # 4. Use cv2.addWeighted() to combine the results
        # 5. Convert each pixel to unint8, then apply threshold to get binary image

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_image, (5,5), 0)
        gX = cv2.Sobel(blur_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        gY = cv2.Sobel(blur_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

        # combined = cv2.addWeighted(gX, 0.9, gY, 0.1, 0)  # 0830
        combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)  # Others

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
        Convert RGB to HSL and threshold to binary image using S channel
        """
        # 1. Convert the image from RGB to HSL
        # 2. Apply threshold on S channel to get binary image

        ##====================== Others ======================##
        white_thres_low = np.array([0, 180, 0], dtype = np.uint8)
        white_thres_up = np.array([180, 255, 255], dtype = np.uint8)

        yellow_thres_low = np.array([20, 0, 90], dtype = np.uint8)
        yellow_thres_up = np.array([30, 255, 255], dtype = np.uint8)
        ##====================================================##

        ##====================== 0830 ======================##
        # white_thres_low = np.array([0, 120, 0], dtype = np.uint8)
        # white_thres_up = np.array([180, 255, 130], dtype = np.uint8)

        # yellow_thres_low = np.array([15, 0, 80], dtype = np.uint8)
        # yellow_thres_up  = np.array([45, 255, 255], dtype = np.uint8)
        ##==================================================##

        img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(img_hls, white_thres_low, white_thres_up)
        yellow_mask = cv2.inRange(img_hls, yellow_thres_low, yellow_thres_up)
        mask = cv2.bitwise_or(white_mask, yellow_mask)

        ##====================== 0484 ======================##
        # gray_thres_low = np.array([0, 0, 0], dtype = np.uint8)
        # gray_thres_up = np.array([180, 5, 210], dtype = np.uint8)

        # white_thres_low = np.array([0, 180, 0], dtype = np.uint8)
        # white_thres_up = np.array([180, 255, 130], dtype = np.uint8)

        # img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # white_mask = cv2.inRange(img_hls, white_thres_low, white_thres_up)
        # yellow_mask = cv2.inRange(img_hls, gray_thres_low, gray_thres_up)
        # mask = cv2.bitwise_or(white_mask, yellow_mask)
        ##==================================================##

        filtered_hls = cv2.bitwise_and(img_hls, img_hls, mask=mask)
        filtered_bgr = cv2.cvtColor(filtered_hls, cv2.COLOR_HLS2BGR)
        filtered_gray = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY)
        filtered_gray[filtered_gray > 0] = 1
        
        return filtered_gray

    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        # 1. Apply sobel filter and color filter on input image
        # 2. Combine the outputs
        grad_thresh_min = 25
        grad_thresh_max = 100
        color_thresh = (100, 255)

        SobelOutput = self.gradient_thresh(img, thresh_min=grad_thresh_min, thresh_max=grad_thresh_max)
        ColorOutput = self.color_thresh(img, thresh=color_thresh)

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput == 1) | (SobelOutput == 1)] = 1
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

        return binaryImage

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        # 1. Visually determine 4 source points and 4 destination points
        # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        # 3. Generate warped image in bird view using cv2.warpPerspective()

        # src_pts = np.float32([[235, 285], [400, 285], [0, 420], [640, 420]])  # Gazebo
        src_pts = np.float32([[500, 230], [770, 230], [200, 360 ], [820, 360]])  # 0011 and 0056
        # src_pts = np.float32([[595, 277], [770, 273], [405, 373 ], [950, 366]])  # 0484
        # src_pts = np.float32([[530, 386], [720, 386], [177, 688 ], [1100, 688]])  # 0830
        dst_pts = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        Minv = np.linalg.inv(M)

        warped_img_size = (img.shape[1], img.shape[0])
        warped_img = cv2.warpPerspective(img.astype("uint8"), M, warped_img_size)

        return warped_img, M, Minv

    def detection(self, img):
        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']
        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True
            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)
                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)