import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
from util import euler_to_quaternion, quaternion_to_euler
import time

class vehicleController():
    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = False

    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp

    # Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    def extract_vehicle_info(self, currentPose):
        pos_x, pos_y, vel, yaw = 0, 0, 0, 0

        pos_x = currentPose.pose.position.x
        pos_y = currentPose.pose.position.y
        vel = math.sqrt((currentPose.twist.linear.x) ** 2 + (currentPose.twist.linear.y) ** 2)
        eulers = quaternion_to_euler(currentPose.pose.orientation.x, currentPose.pose.orientation.y, currentPose.pose.orientation.z, currentPose.pose.orientation.w)
        yaw = eulers[-1]

        return pos_x, pos_y, vel, yaw # note that yaw is in radian

    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):
        target_velocity = 12
        tol_angle = (30 * math.pi) / 180
        # Get future waypoints curvature
        g1_x = future_unreached_waypoints[0][0]
        g1_y = future_unreached_waypoints[0][1]

        g2_x = g1_x
        g2_y = g1_y
        if(len(future_unreached_waypoints) >= 2):
            g2_x = future_unreached_waypoints[1][0] 
            g2_y = future_unreached_waypoints[1][1]

        g3_x = g1_x
        g3_y = g1_y
        if(len(future_unreached_waypoints) >= 3):
            g3_x = future_unreached_waypoints[2][0] 
            g3_y = future_unreached_waypoints[2][1]
    
        angle1 = math.atan2(g1_y - g2_y, g1_x - g2_x)
        angle2 = math.atan2(g2_y - g3_y, g2_x - g3_x)
        road_curvature = angle1 - angle2

        # Get the heading angle between vehicle's position and the first waypoint
        # Transformation from world to body frame
        g1_x_t = g1_x - curr_x
        g1_y_t = g1_y - curr_y
        R_inv = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                          [-np.sin(curr_yaw), np.cos(curr_yaw)]])
        g1_body = np.dot(R_inv, np.array([g1_x_t, g1_y_t]))
        g1_x_body = g1_body[0]
        g1_y_body = g1_body[1]
        goal_angle = math.atan2(g1_y_body, g1_x_body)
        
        target_velocity -= (0.5 * abs(goal_angle) + abs(road_curvature)) * (4 / tol_angle)
        target_velocity = max(target_velocity, 8)

        return target_velocity

    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):
        target_steering = 0

        # Average the position between the first two unreached waypoints. Choose it as the lookahead point
        fake_target = target_point
        if(len(future_unreached_waypoints) > 1):
            fake_target = [(2*future_unreached_waypoints[0][0] + future_unreached_waypoints[1][0] )/3, (2*future_unreached_waypoints[0][1] + future_unreached_waypoints[1][1] )/3]
        ld = math.sqrt((fake_target[0] - curr_x) ** 2 + (fake_target[1] - curr_y) ** 2)

        # Solve for alpha
        fake_x_t = fake_target[0] - curr_x
        fake_y_t = fake_target[1] - curr_y
        R_inv = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                          [-np.sin(curr_yaw), np.cos(curr_yaw)]])	
        fake_body = np.dot(R_inv, np.array([fake_x_t, fake_y_t]))
        fake_x_body = fake_body[0]
        fake_y_body = fake_body[1]
        alpha = math.atan2(fake_y_body, fake_x_body)

        target_steering = math.atan2(2 * self.L * math.sin(alpha), ld)
        
        return target_steering


    def execute(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz

        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)

        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.controlPub.publish(newAckermannCmd)
