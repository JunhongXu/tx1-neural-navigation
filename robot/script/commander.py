#!/usr/bin/env python
"""
Subscriber:
    1. Commander node subscribes to /joy topic published by the host pc and listens to the command of joystick and
       send commands to /cmd_vel.
    2. Commander node subscribes to /neural_command topic published by neural_commander node to send commands to
       /cmd_vel
Publisher:
    1. Commander node publishes Twist to /cmd_vel topic to control the icreate robot
"""


import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped, Twist
from controller import *


class Commander(object):
    def __init__(self):
        rospy.loginfo('[*]Start commander.')
        # joystick
        self.joycmd = Twist()
        self.nn_cmd = Twist()
        # mode
        self.neuralnet_mode = False
        self.ps3 = PS3()
        # subscriber
        rospy.Subscriber('/joy', Joy, self.joystick_cmd, queue_size=5)
        rospy.Subscriber('/neural_cmd', Twist, self.neural_cmd, queue_size=5)
        # publisher
        self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # initialize rosnode
        rospy.init_node('commander', anonymous=True)
        self.rate = rospy.Rate(60)

    def run(self):
        while not rospy.is_shutdown():
            self.send_cmd()
            self.rate.sleep()

    def joystick_cmd(self, cmd):
        """joystick command, -0.5<=linear.x<=0.5; -4.25<=auglar.z<=4.25"""
        self.ps3.update(cmd)
        vel = self.ps3.left_stick*0.5
        angular = self.ps3.right_stick * 4.25
        self.joycmd.angular.z = angular
        self.joycmd.linear.x = vel

        btn_events = self.ps3.btn_events
        if 'R2_pressed' in btn_events:
            self.neuralnet_mode = not self.neuralnet_mode
            if self.neuralnet_mode:
                rospy.loginfo('[*]Neural network controlling...')
            else:
                rospy.loginfo('[*]Human controlling...')

    def neural_cmd(self, cmd):
        self.nn_cmd = cmd

    def send_cmd(self):
        if self.neuralnet_mode:
            self.move_pub.publish(self.nn_cmd)
        else:
            self.move_pub.publish(self.joycmd)


if __name__ == '__main__':
    try:
        commander = Commander()
        commander.run()
    except rospy.ROSInterruptException:
        pass
