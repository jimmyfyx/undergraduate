# ECE470 Final Project (Lab5)  
## Completed by Yixiao Fang (yixiaof2@illinois.edu) and Josh Akin (jdakin2@illinois.edu)
## April, 2022 - May, 2022
The task for this project is to find and pick colored blocks with the UR3 robot arm, and move the blocks to the target position within the robot's workspace. A more detailed introduction is in the *Lab5_Manual.pdf* file.<br/>
Techniques used in this project:
- ROS (Robot Operating System)
- OpenCV Package
- Forward Kinematics
- Inverse Kinematics

## Commands Lines to run the code
### Terminal 1  
#### $ cd ~/catkin_NETID  
#### $ catkin_make  
#### $ source devel/setup.bash  
#### $ roslaunch ur3_driver ur3_vision_driver.launch  

### Terminal 2  
#### $ source devel/setup.bash  
#### $ rosrun lab5pkg_py lab5_exec.py
