import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
from model.tf_model import NeuralCommander
import os


def __split_name(labels, name):
    splitted = name.split('_')
    v = float(splitted[-2])
    r = float(splitted[-1].strip('.png'))
    label = np.array([v/0.5, r/4.25], dtype=np.float32)
    labels.append(label)


def load_data(iteration, val_num=200, read_rgb=True, read_depth=False, display=False, safety=False):
    """Read all images in RGB and DEPTH"""
    rgb_imgs = []
    rgb_labels = []
    depth_imgs = []
    depth_labels = []
    filedir = '/media/jxu7/BACK-UP/Data/neural-navigation/iteration_%s' % iteration
    filedir = os.path.join(filedir, 'safety' if safety else 'primary')
    print(filedir)

    if read_rgb:
        rgb_names = glob.glob(os.path.join(filedir, 'RGB_DATA/*.png'))
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
        depth_names = glob.glob(os.path.join(filedir, 'DEPTH_DATA/*.png'))
        print('[*]Collected %s DEPTH pictures.' % len(depth_names))
        for n in depth_names:
            __split_name(depth_labels, n)
            depth_img = cv2.imread(n)
            depth_img = cv2.resize(depth_img, (128, 128))
            depth_imgs.append(depth_img)
    return np.array(rgb_imgs), np.array(rgb_labels), np.array(depth_imgs), np.array(depth_labels)


def convert_to_pkl(train_iter):
    """Save tensorflow model to a pickle file"""
    params = {}
    with tf.Session() as sess:
        model = NeuralCommander()
        model.restore(sess, train_iter)
        for v in model.params:
            params[v.name] = sess.run(v)
    with open('../checkpoint/%s/pkl_model.pkl' % train_iter, 'w') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
    print('[*]Saved to pkl file')
