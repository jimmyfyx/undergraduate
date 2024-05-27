import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode

import random
import bisect

def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        particles = list()

        # Modify the initial particle distribution to be within the top-right quadrant of the world, and compare the performance with the whole map distribution.
        for i in range(num_particles):
            # (Default) The whole map
            # x = np.random.uniform(0, world.width)
            # y = np.random.uniform(0, world.height)
            # heading = random.uniform(0, 2 * np.pi)

            ## first quadrant
            x = random.uniform(world.width/2, world.width)
            y = random.uniform(world.height/2, world.height)
            heading = random.uniform(0, 2 * np.pi)

            particles.append(Particle(x = x, y = y, heading=heading, maze = world, sensor_limit = sensor_limit))

        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle
        self.control_input_len = 0
        return

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """

        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self,x1, x2, std = 5000):
        if x1 is None: # If the robot recieved no sensor measurement, the weights are in uniform distribution.
            return 1./len(self.particles)
        if x2 is None: # If the particle recieved no sensor measurement, the weights are in uniform distribution.
            return 1./len(self.particles)
        else:
            tmp1 = np.array(x1)
            tmp2 = np.array(x2)
            return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))

    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """
        sum_weight = 0
        for particle in self.particles:
            particle_reading = particle.read_sensor()
            fake_weight = self.weight_gaussian_kernel(readings_robot, particle_reading, std=5000)
            sum_weight += fake_weight
            particle.weight = fake_weight
        # Normalize weights
        for particle in self.particles:
            particle.weight = particle.weight / sum_weight
        return

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = [ ]

        n = self.num_particles
        weights = []
        for i in range(n):
            weights.append(self.particles[i].weight)
        weights = np.array(weights)
        cumsum_weights = np.cumsum(weights)

        for i in range(n):
            index = bisect.bisect(cumsum_weights, random.random())
            particles_new.append(Particle(x=self.particles[index].x, y=self.particles[index].y, heading=self.particles[index].heading, maze=self.world, sensor_limit=self.sensor_limit, noisy=True))

        self.particles = particles_new
        return
    
    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
        """       
        for i in range(self.control_input_len, len(self.control)):
            velocity = self.control[i][0]
            steering_angle = self.control[i][1]
            dt = 0.01
            for particle in self.particles:
                arr = vehicle_dynamics(0, [particle.x, particle.y, particle.heading], velocity, steering_angle)
                particle.x += arr[0] * dt
                particle.y += arr[1] * dt
                particle.heading += arr[2] * dt  
        self.control_input_len = len(self.control)

    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        error = []
        count = 200
        while True:
            count -= 1

            if(count == 0):
                print(error)

            self.particleMotionModel()
            reading = self.bob.read_sensor()
            self.updateWeight(reading)
            self.resampleParticle()
            
            self.world.show_robot(robot=self.bob)
            info_list = self.world.show_estimated_location(particles=self.particles)
            self.world.show_particles(particles=self.particles)
            self.world.clear_objects()

            robot_x = self.bob.x
            robot_y = self.bob.y
            avg_x = info_list[0]
            avg_y = info_list[1]
            error.append(np.sqrt((robot_x - avg_x) ** 2 + (robot_y - avg_y) ** 2))
            ###############