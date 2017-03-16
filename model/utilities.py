import numpy as np
import cv2
import glob


def load_data(val_num=200, read_rgb=True, read_depth=False, display=False):
    """Read all images in RGB and DEPTH"""
    def __split_name(labels, name):
        # don't know why these are needed....
        splitted = name.strip(' ').split('-')
        # get velocity
        try:
            v = float(splitted[1])
        except:
            v = -float(splitted[2])
            # get rotation
        if splitted[-2] == '':
            r = -float(splitted[-1].strip('.png'))
        else:
            r = float(splitted[-1].strip('.png'))
        label = np.array([v/0.5, r/4.25], dtype=np.float32)
        print(label)
        labels.append(label)

    rgb_imgs = []
    rgb_labels = []
    depth_imgs = []
    depth_labels = []

    if read_rgb:
        rgb_names = glob.glob('RGB_DATA/*.png')
        print('[*]Collected %s RGB pictures.' % len(rgb_names))
        for n in sorted(rgb_names):
            __split_name(rgb_labels, n)
            rgb_img = cv2.imread(n)
            if display:
                cv2.imshow('test', rgb_img)
                cv2.waitKey(5)
            rgb_img = cv2.resize(rgb_img, (128, 128))
            rgb_imgs.append(rgb_img)

    if read_depth:
        depth_names = glob.glob('DEPTH_DATA/*.png')
        print('[*]Collected %s DEPTH pictures.' % len(depth_names))
        for n in depth_names:
            __split_name(depth_labels, n)
            depth_img = cv2.imread(n)
            depth_img = cv2.resize(depth_img, (128, 128))
            depth_imgs.append(depth_img)
    return np.array(rgb_imgs), np.array(rgb_labels), np.array(depth_imgs), np.array(depth_labels)
