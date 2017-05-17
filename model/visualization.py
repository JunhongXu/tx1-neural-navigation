from model.tf_model import NeuralCommander
import tensorflow as tf
import numpy as np
from scipy.misc import imsave, imread, imresize
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import conv2d_transpose
from utilities import load_data


def extract_feat_maps(model, x, save_dir, index, sess):
    """
    This function saves the averaged feature maps of given image x to a directory.
    """
    if len(x.shape) == 3:
        x = np.expand_dims(x, 0)

    # sess.run(tf.global_variables_initializer())
    layers = model.layers[:7]
    # let's first do a inference to see what is the predicted value
    # print(sess.run(model.pi, feed_dict={model.x: x, model.is_training: False}))
    features = sess.run([layer for layer in layers[::2]], feed_dict={model.x: x})
    features = [np.mean(feature, axis=3) for feature in features]

    for i, feature in enumerate(features):
        imsave(arr=np.squeeze(feature), name=os.path.join(save_dir, 'conv{}/image_{}.jpg'.format(i+1, index)))
    return features


def build_deconv():
    feature0 = tf.placeholder(dtype=tf.float32, shape=[None, 16, 16, 1])
    feature1 = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1])
    feature2 = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    feature3 = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1])
    w_init = tf.ones_initializer()
    b_init = tf.zeros_initializer()
    with tf.variable_scope('visualization'):
        deconv1 = conv2d_transpose(feature0, 1, kernel_size=3, stride=2, weights_initializer=w_init,
                                   biases_initializer=b_init, padding='SAME', scope='deconv1')
        combined_feat1 = tf.multiply(deconv1, feature1)
        deconv2 = conv2d_transpose(combined_feat1, 1, kernel_size=3, stride=2, weights_initializer=w_init,
                                   biases_initializer=b_init, padding='SAME', scope='deconv2')
        combined_feat2 = tf.multiply(deconv2, feature2)
        deconv3 = conv2d_transpose(combined_feat2, 1, kernel_size=3, stride=2, weights_initializer=w_init,
                                   biases_initializer=b_init, padding='SAME', scope='deconv3')
        combined_feat3 = tf.multiply(feature3, deconv3)

        # scale
        min_val = tf.reduce_min(combined_feat3)
        max_val = tf.reduce_max(combined_feat3)
        combined_feat3 = (combined_feat3 - min_val)/(max_val - min_val)
    return feature0, feature1, feature2, feature3, combined_feat3


def upsample_multiply(feature, f0, f1, f2, f3, combined, index, sess):
    """
    First build a graph of three deconv layers. The inputs are feature_maps[0], feature_maps[0]*feature_maps[1]...
    Then save the final result

    feature is a list of 4 numpy arrays of shape (N, H, W, 1)
    """


    # for i in range()
    final_feature = sess.run(combined, feed_dict={
            f0: np.expand_dims(feature[-1], axis=-1),
            f1: np.expand_dims(feature[-2], axis=-1),
            f2: np.expand_dims(feature[-3], axis=-1),
            f3: np.expand_dims(feature[-4], axis=-1)
        })
    imsave('../visualization_imgs/final/image_{}.jpg'.format(index), np.squeeze(final_feature))
    return final_feature


def overlay(x, visual_map, index):
    """
    x: H, W, C
    visual_map: N, H, W, 1
    """
    feature = visual_map[0]
    idx = np.where(np.squeeze(feature) >= 0.1)
    x[idx] = [255, 0, 0] * (1- x[idx])
    imsave('../visualization_imgs/overlayed/image_{}.jpg'.format(index), x)
    cv2.imshow('feature', feature)
    cv2.imshow('image', x)
    cv2.waitKey(30)


if __name__ == '__main__':
    nn = NeuralCommander()
    save_dir = '../visualization_imgs'
    image, _, _ , _ = load_data(1)
    f0, f1, f2, f3, combined = build_deconv()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nn.restore(sess, 1)
        for index, i in enumerate(image):
            f = extract_feat_maps(nn, i, save_dir, index, sess)
            feat = upsample_multiply(f, f0, f1, f2, f3, combined, index, sess)
            # print(np.where(feat >= 0.05))
            overlay(i, feat, index)
