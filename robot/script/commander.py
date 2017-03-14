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
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped, Twist
import numpy as np


class Commander(object):
    def __init__(self):
        # joystick
        self.joycmd = TwistStamped()
        # neural net
        self.nncmd = TwistStamped()
        # mode
        self.mode = 'human'
        # subscriber
        rospy.Subscriber('/joy', Joy, self.joystick_cmd, queue_size=5)
        rospy.Subscriber('/neural_command', TwistStamped, self.neural_cmd)

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
        print('here')
        vel = cmd.axes[1]/2.
        angular = cmd.axes[2] * 4.25
        self.joycmd.twist.angular.z = angular
        self.joycmd.twist.linear.x = vel

    def neural_cmd(self):
        pass

    def send_cmd(self):
        if self.mode == 'human':
            cmd = self.joycmd.twist
            rospy.loginfo('human command: {}, {}'.format(cmd.linear.x, cmd.angular.z))
        elif self.mode == 'neural':
            cmd = self.nncmd.twist
            rospy.loginfo('neural command: {}, {}'.format(cmd.linear.x, cmd.angular.z))
        self.move_pub.publish(cmd)



if __name__ == '__main__':
    try:
        commander = Commander()
        commander.run()
    except rospy.ROSInterruptException:
        pass
