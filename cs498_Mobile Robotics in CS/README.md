# CS498 Mobile Robotics in Computer Science (Fall 2022)
This course is about principles in mobile robotics, including basic kinematics, dynamics, control, and perception. For every module in this course, there is a corresponding problem set or coding exercise. <br/>

- All the coding exercises are completed by ROS2, so the related executable files are in the ROS2 workspace. <br/>
- There are also problem sets (also programming assignments), included in the 'Problem Sets' folder. <br/>
- For every coding exercise or problem set, there is a pdf file containing all the instructions. <br/>

More information please refer to the course website: http://daslab.illinois.edu/coursesfall2021.html

**11/9/2022 Update** coding exercise 1 relevant files added in the ROS2 workspace (course in progress) <br/>
**12/28/2022 Update** reupload the ROS2 workspace, including coding exercise 3 and final project relevant files

## Instructions to build the complete ROS2 workspace
**Due to the file size limit of GitHub, some files in the original ROS2 workspace are not uploaded, so there are few additional steps to take to build the complete ROS2 workspace.**

Inside the folder *cs498_ros2ws*, there only exists the *src* folder of the workspace. To build the complete workspace, a new ROS2 workspace is needed, then put the package *mobile_robotics* into the *src* folder in the new workspace. Finally, build the new workspace to finish.

In addition, there are some other requirements that need to be installed to run the scripts: <br/>
- To run the script of coding exercise 3, the *lidar* rosbag file is required (https://drive.google.com/drive/folders/1j3mt22w97_7BYBUkn2Kr3sK8uN2--6g9?usp=share_link). Download the files and put them in a folder called *lidar*, then put the *lidar* folder in anywhere in the workspace you like. Also, follow the instructions in *2022_Coding exercise 3.pdf* to install Gmapping for ROS2. <br/>
- To run the script of rtabmap exercise (final project), follow the instructions in *2022_Final project.pdf* to install two rosbag files and the rtabmap package for ROS2 (https://github.com/introlab/rtabmap_ros/tree/ros2#rtabmap_ros).
