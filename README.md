# Navigation in indoor environments using iCreate robot with TX1 + ZED Camera

## Pre-requisite

Initialize a catkin workspace in TX1:

`mkdir -p catkin_ws/src`

`cd catkin_ws/src`

`catkin_init_workspace`

`cd ..`

`catkin_make`

Source the workspace

`source devel/setup.bash`

Clone the packages from to catkin_ws/src:

1. create_autonomy packages, can be found at 
https://github.com/autonomylab/create_autonomy

2. zed-ros-wrapper, can be found at
https://github.com/stereolabs/zed-ros-wrapper

3. neural-navigation, https://github.com/JunhongXu/tx1-neural-navigation

In catkin_ws, do `catkin_make`.

## Host PC (Assume you have a PC3 controller connected to Host PC)

On host PC, run 

1. `roscore`

2. `rosrun joy joy_node`


## TX1

On TX1, run `roslaunch robot run_robot.launch`


