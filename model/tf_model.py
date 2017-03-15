import tensorflow as tf
from tensorflow.contrib import layers

# check the version of tensorflow
tf_version = tf.__version__
if tf_version.split('.')[0] < 0:
    older = True
else:
    oler = False

class NeuralCommander(object):
    def __init__(self, sess, inpt_size=(128, 128, 3)):
        self.sess = sess
        self.x = tf.placeholder(shape=(None, ) + inpt_size, name='image', dtype=tf.float32)
        # two elements, first one is velocity, last one is rotation
        self.y = tf.placeholder(shape=(None, 2), name='twist', dtype=tf.float32)

        self.layers = []

        # build the model
        self.pi = self.build()

        # loss
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(self.y, self.pi), axis=1)), name='loss')

        # initialize all variables
        if older:
            self.sess.run(tf.initialize_all_variables())
        else:
            self.sess.run(tf.global_variables_initializer())

    def build(self):
        with tf.name_scope('normalize'):
            image = tf.div(self.x, 255)

        with tf.variable_scope('conv1') as scope:
            conv1 = layers.conv2d(image, 64, 3, scope=scope)
            self.layers.append(conv1)
        with tf.variable_scope('pool1') as scope:
            pool1 = layers.max_pool2d(conv1, kernel_size=2, padding='SAME', scope=scope)
            self.layers.append(pool1)

        with tf.variable_scope('conv2') as scope:
            conv2 = layers.conv2d(pool1, num_outputs=128, kernel_size=3, scope=scope)
            self.layers.append(conv2)
        with tf.variable_scope('pool2') as scope:
            pool2 = layers.max_pool2d(conv2, kernel_size=2, padding='SAME', scope=scope)

        with tf.variable_scope('conv3') as scope:
            conv3 = layers.conv2d(pool2, num_outputs=128, kernel_size=3, scope=scope)
        with tf.variable_scope('pool3') as scope:
            pool3 = layers.max_pool2d(conv3, kernel_size=2, padding='SAME', scope=scope)

