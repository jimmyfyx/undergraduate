import math
import numpy as np
import rclpy
from rclpy.node import Node

from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped, TransformStamped

class Rtabmap(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subs_ekf = self.create_subscription(Odometry, 'terrasentia/ekf', self.callback_ekf, 10)
        self.subs_odom = self.create_subscription(Odometry, 'rtabmap/odom', self.callback_odom, 10)

        # ekf position ('base_link' frame) is with respect to 'map' frame
        self.ekf_x = 0.0
        self.ekf_y = 0.0
        self.ekf_z = 0.0
        self.ekf_qx = 0.0
        self.ekf_qy = 0.0
        self.ekf_qz = 0.0
        self.ekf_qw = 0.0
        # rtabmap estimated position ('zed2_imu_link' frame) is with respect to 'odom' frame
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_z = 0.0
        self.odom_qx = 0.0
        self.odom_qy = 0.0
        self.odom_qz = 0.0
        self.odom_qw = 0.0

        self.current_time = 0.0

        # record trajectories
        self.ekf_tra_file = open("ekf_tra.txt", "w+")
        self.rtabmap_tra_file = open("rtabmap_tra.txt", "w+")
        self.timer = self.create_timer(0.1, self.callback_write_ekf)
        self.timer_1 = self.create_timer(0.1, self.callback_write_rtabmap)
    
    def callback_ekf(self, msg):
        # get ground-truth robot position and orientation
        self.ekf_x = msg.pose.pose.position.x
        self.ekf_y = msg.pose.pose.position.y
        self.ekf_z = msg.pose.pose.position.z
        self.ekf_qx = msg.pose.pose.orientation.x
        self.ekf_qy = msg.pose.pose.orientation.y
        self.ekf_qz = msg.pose.pose.orientation.z
        self.ekf_qw = msg.pose.pose.orientation.w

    def callback_odom(self, msg):
        # get rtabmap estimated robot position and orientation
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_z = msg.pose.pose.position.z
        self.odom_qx = msg.pose.pose.orientation.x
        self.odom_qy = msg.pose.pose.orientation.y
        self.odom_qz = msg.pose.pose.orientation.z
        self.odom_qw = msg.pose.pose.orientation.w
    
    def callback_write_ekf(self):
        # save ground-truth trajectory
        self.current_time = self.get_clock().now().nanoseconds/1e9
        self.ekf_tra_file.write(str(self.current_time) + " " + str(self.ekf_x) + " " + str(self.ekf_y) + " " + str(self.ekf_z) + " " + str(self.ekf_qx) + " " + str(self.ekf_qy) + " " + str(self.ekf_qz) + " " + str(self.ekf_qw) + '\n')
    
    def callback_write_rtabmap(self):
        # save estimated trajectory
        self.current_time = self.get_clock().now().nanoseconds/1e9
        self.rtabmap_tra_file.write(str(self.current_time) + " " + str(self.odom_x) + " " + str(self.odom_y) + " " + str(self.odom_z) + " " + str(self.odom_qx) + " " + str(self.odom_qy) + " " + str(self.odom_qz) + " " + str(self.odom_qw) + '\n')
        


def main(args=None):
    rclpy.init(args=args)

    rtabmap_node = Rtabmap()

    rclpy.spin(rtabmap_node)
    rtabmap_node.ekf_tra_file.close()
    rtabmap_node.rtabmap_tra_file.close()
    rtabmap_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()