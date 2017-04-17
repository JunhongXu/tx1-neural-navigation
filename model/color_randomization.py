from pylab import *
import cv2
import numpy as np


def color_randomization(img, p):
    """img is a numpy array of shape (N, H, W, C). p is the ratio to randomize pictures"""
    N, H, W, C = img.shape
    random_index = np.random.randint(0, N, int(N*p))
    phi = np.random.randint(40, 50)
    theta = np.random.randint(10, 15)
    maxIntensity = 255
    img[random_index] = (maxIntensity/phi)*(img[random_index]/(maxIntensity/theta))**2
    # img = cv2.GaussianBlur(img,(3, 3),0)

    # define range of blue color in HSV
    # lower_door = np.array([50, 60, 35])
    # upper_door = np.array([175, 130, 110])
    # mask = cv2.inRange(hsv, lower_door, upper_door)
    # img[np.where(mask==255)]=np.array([0, 50, 255])
    #
    # lower_wall = np.array([90, 1, 80])
    # upper_wall = np.array([175, 30, 130])
    # mask = cv2.inRange(hsv, lower_wall, upper_wall)
    # img[np.where(mask==255)] = np.array([125, 125, 125])
    # res = cv2.bitwise_and(img, img, mask=mask)
    # imshow(img)
    return img
