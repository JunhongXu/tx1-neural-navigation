import tensorflow as tf
from model.tf_model import NeuralCommander
from model.utilities import load_data

# check the version of tensorflow
tf_version = tf.__version__
if tf_version.split('.')[0] < 0:
    older = True
else:
    older = False


def train(sess, model, data):
    pass


if __name__ == '__main__':
    with tf.Session() as sess:
        model = NeuralCommander()
        data = load_data()
        # initialize all variables
        if older:
            sess.run(tf.initialize_all_variables())
        else:
            sess.run(tf.global_variables_initializer())

        train(sess, model, data)

