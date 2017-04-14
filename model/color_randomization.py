import numpy as np
from pylab import *
from matplotlib import pyplot as plt
import cv2

filepath = '/media/jxu7/BACK-UP/Data/neural-navigation/primary/RGB_DATA/1/1491954394965768562_0.5_-0.0.png'

phi = 50
theta = 10

def color_randomization():
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    # img = img.astype(np.int32)
    print(img)
#    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#    print(hsv)
    H, W, C = img.shape
    maxIntensity = 255
    # img[:,:,1] += np.random.randint(0, np.random.randint(1, 50), (H, W), dtype=np.uint8)
    # img[:,:,2] += np.random.randint(0, np.random.randint(1, 50), (H, W), dtype=np.uint8)
    # img[:,:,0] += np.random.randint(0, np.random.randint(1, 50), (H, W), dtype=np.uint8)
    kernel = np.ones((3,3),np.float32)/25
    img = (maxIntensity/phi)*(img/(maxIntensity/theta))**2
    img = cv2.GaussianBlur(img,(3,3),0)

    print(np.max(img))
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

    imshow(img)
    plt.show()


if __name__ == '__main__':
    color_randomization()
