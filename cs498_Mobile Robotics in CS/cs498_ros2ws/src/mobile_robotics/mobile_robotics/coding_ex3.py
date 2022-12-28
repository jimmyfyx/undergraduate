# Student name: 

import math
import numpy as np
import rclpy
from rclpy.node import Node

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped, TransformStamped
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
import matplotlib.pyplot as plt
import time
from tf2_msgs.msg import TFMessage
from copy import copy
from visualization_msgs.msg import Marker

# Further info:
# On markers: http://wiki.ros.org/rviz/DisplayTypes/Marker
# Laser Scan message: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html

class CodingExercise3(Node):
    def __init__(self):
        super().__init__('CodingExercise3')

        self.ranges = [] # lidar measurements
        
        self.point_list_1 = [] # A list of points to draw lines
        self.line = Marker()
        self.line_marker_init(self.line)

        # robot pose
        self.robo_x = 0
        self.robo_y = 0
        self.robo_yaw = 0

        # Ros subscribers and publishers
        self.subscription_ekf = self.create_subscription(Odometry, 'terrasentia/ekf', self.callback_ekf, 10)
        self.subscription_scan = self.create_subscription(LaserScan, 'terrasentia/scan', self.callback_scan, 10)
        self.pub_lines = self.create_publisher(Marker, 'lines', 10)
        self.timer_draw_line_example = self.create_timer(0.1, self.draw_line_example_callback)

    
    def callback_ekf(self, msg):
        # Read the translation and rotation of the robot with respect to the odometry frame
        self.robo_x = msg.pose.pose.position.x
        self.robo_y = msg.pose.pose.position.y
        quar = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z])
        _, _, self.robo_yaw = self.quat2euler(quar)

    
    def callback_scan(self, msg):
        self.ranges = list(msg.ranges) # Lidar measurements
        self.ranges = np.array(self.ranges)
        # clip the ranges to make visualization cleaner
        self.ranges = np.clip(self.ranges, a_min=0.0, a_max=7.0)
        print("some-ranges:", self.ranges[0:5])
        print("Number of ranges:", len(self.ranges))

        
    def quat2euler(self, q): # q, as np.array() [qw, qx, qy, qz]
        q0, q1, q2, q3 = q.squeeze().tolist()

        m=np.eye(3,3)
        m[0,0] = 1.0 - 2.0*(q2*q2 + q3*q3)
        m[0,1] = 2.0*(q1*q2 - q0*q3)
        m[0,2] = 2.0*(q1*q3 + q0*q2)
        m[1,0] = 2.0*(q1*q2 + q0*q3)
        m[1,1] = 1.0 - 2.0*(q1*q1 + q3*q3)
        m[1,2] = 2.0*(q2*q3 - q0*q1)
        m[2,0] = 2.0*(q1*q3 - q0*q2)
        m[2,1] = 2.0*(q2*q3 + q0*q1)
        m[2,2] = 1.0 - 2.0*(q1*q1 + q2*q2)
        phi = math.atan2(m[2,1], m[2,2])
        theta = -math.asin(m[2,0])
        psi = math.atan2(m[1,0], m[0,0])
        return phi, theta, psi
    

    '''
    Helper functions to fit lines
    '''
    def polar_to_xy(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    
    def estimate_coef(self, x, y):
        b_1 = (y[0] - y[1]) / (x[0] - x[1])
        b_0 = y[0] - b_1 * x[0]
  
        return (b_0, b_1)

    
    def shortest_distance(self, x1, y1, a, b, c):
        d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
        return d


    def draw_line_example_callback(self):
        if len(self.ranges) != 0:
            x_pos = self.robo_x
            y_pos = self.robo_y
            yaw_angle = self.robo_yaw
            # lidar data is not in the defined robot local frame. Need a rotation matrix to transform into local frame
            R_to_local = np.array([[np.cos(-1.5708), -np.sin(-1.5708)], [np.sin(-1.5708), np.cos(-1.5708)]])
            R_to_global = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle)], [np.sin(yaw_angle), np.cos(yaw_angle)]])

            # convert polar coordinates to xy coordinates
            rho = np.array(self.ranges)
            theta = (math.pi/180) * np.linspace(-45, 225, 1081)
            x, y = self.polar_to_xy(rho, theta)

            '''
            Split and Merge algorithm
            '''
            # initialize the first set to contain all the points
            first_set_x = x[60:1050]
            first_set_y = y[60:10500]
            
            # initialize the queue
            points_x_queue = []
            points_y_queue = []
            points_x_queue.append(first_set_x)
            points_y_queue.append(first_set_y)

            self.point_list_1 = [] # comment this line if all the published lines need to be seen

            dist_thres = 0.2
            while len(points_x_queue) != 0 and len(points_y_queue) != 0:
                cur_set_x = points_x_queue.pop(0)
                cur_set_y = points_y_queue.pop(0)

                if len(cur_set_x) > 2:
                    # fit the line for two extreme points6
                    x1 = cur_set_x[0]
                    y1 = cur_set_y[0]
                    x2 = cur_set_x[cur_set_x.shape[0] - 1]
                    y2 = cur_set_y[cur_set_y.shape[0] - 1]
                    x_ext = np.array([x1, x2])
                    y_ext = np.array([y1, y2])

                    b = self.estimate_coef(x_ext, y_ext)  # b[0] is b, b[1] is the slope

                    # find the greatest distance to the line
                    max_dist = -1
                    split_idx = -1
                    for i in range(cur_set_x.shape[0]):
                        dist = self.shortest_distance(cur_set_x[i], cur_set_y[i], b[1], -1, b[0])
                        if dist > max_dist:
                            max_dist = dist
                            split_idx = i
                    
                    # determine whether to split the set
                    if max_dist > dist_thres:
                        # split the set
                        subset_1_x = cur_set_x[0:split_idx + 1]
                        subset_1_y = cur_set_y[0:split_idx + 1]
                        subset_2_x = cur_set_x[split_idx:]
                        subset_2_y = cur_set_y[split_idx:]
                        
                        points_x_queue.append(subset_1_x)
                        points_y_queue.append(subset_1_y)
                        
                        points_x_queue.append(subset_2_x)
                        points_y_queue.append(subset_2_y)
                    else:
                        # the line is good, store line
                        p0 = Point()
                        # convert the point to local frame then global frame
                        local_xyz = np.matmul(R_to_local, np.array([[x_ext[0]], [y_ext[0]]]))
                        global_xyz = np.matmul(R_to_global, np.array([[local_xyz[0][0]], [local_xyz[1][0]]])) + np.array([[x_pos], [y_pos]])
                        p0.x = local_xyz[0][0]
                        p0.y = local_xyz[1][0]
                        p0.z = 0.0
                        
                        p1 = Point()
                        # convert the point to local frame then global frame
                        local_xyz = np.matmul(R_to_local, np.array([[x_ext[1]], [y_ext[1]]]))
                        global_xyz = np.matmul(R_to_global, np.array([[local_xyz[0][0]], [local_xyz[1][0]]])) + np.array([[x_pos], [y_pos]])
                        p1.x = local_xyz[0][0]
                        p1.y = local_xyz[1][0]
                        p1.z = 0.0
                        
                        # make sure horizontal lines are not shown by not publishing points far apart 
                        if abs(x_ext[1] - x_ext[0]) < 1.3:
                            self.point_list_1.append(copy(p0))
                            self.point_list_1.append(copy(p1))
                            self.line.points = self.point_list_1
                            self.pub_lines.publish(self.line)
        
        
    def line_marker_init(self, line):
        line.header.frame_id="/odom"
        line.header.stamp=self.get_clock().now().to_msg()

        line.ns = "markers"
        line.id = 0

        line.type=Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0

        line.scale.x = 0.05
        line.scale.y= 0.05
        
        line.color.r = 1.0
        line.color.a = 1.0
        #line.lifetime = 0


def main(args=None):
    rclpy.init(args=args)

    cod3_node = CodingExercise3()
    
    rclpy.spin(cod3_node)

    cod3_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
