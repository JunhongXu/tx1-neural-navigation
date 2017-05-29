import tensorflow as tf
from model.tf_model import NeuralCommander
from model.color_randomization import *
from model.utilities import *
from tensorflow.contrib.layers import optimize_loss
import numpy as np


TRAIN_ITER = 4
BATCH_SIZE = 128
SAFETY_THRESHOLD = 0.0005
DISPLAY = False
NUM_ITERS = 25000
RANDOMIZE = True
P = 0.3


def train_primary_policy(sess, x, y, writer, model, num_iter, trainer):
    N, H, W, C = x.shape
    # grad_summ = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=)
    grad_summ = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='primary_optimizer')
    summary = tf.summary.scalar('primary_loss', model.loss)
    summary = tf.summary.merge([summary] + grad_summ)
    # train the primary policy
    for i in range(num_iter):
        # random index
        index = np.random.randint(0, N, size=BATCH_SIZE)
        if RANDOMIZE:
            data = color_randomization(x[index], P)
        else:
            data = x[index]

        loss, _, loss_summary, pi = sess.run([model.loss, trainer, summary, model.pi], feed_dict={model.x: data,
                                                             model.y: y[index], model.is_training:True})
        writer.add_summary(loss_summary, global_step=i)
        if loss == np.nan:
            print(y[index])
        if i % 99 == 0:
            # save
            model.save(sess, TRAIN_ITER)
            print('[*]At iteration %s/%s, loss is %s' % (i, num_iter, loss))
            print('[*]Ground Truth %s / Predicted %s' % (y[index][0], pi[0]))


def train_safety_policy(sess, safety_x, pi_label, writer, model, num_iter, trainer):

    N, H, W, C = safety_x.shape
    summary = tf.summary.scalar('safety_loss', model.safety_loss)
    x, y = convert_labels(sess=sess, model=model, reference_label=pi_label, safe_img=safety_x,
                          threshhold=SAFETY_THRESHOLD, randomize=RANDOMIZE, p=P)
    # train the safety policy
    for i in range(num_iter):
        # random index
        index = np.random.randint(0, N, size=BATCH_SIZE)
        loss, _, loss_summary = sess.run([model.safety_loss, trainer, summary], feed_dict={model.safety_inpt: x[index],
                                                             model.safety_y: y[index], model.is_training:True})
        writer.add_summary(loss_summary, global_step=i)
        if i % 99 == 0:
            # save
            model.save(sess, TRAIN_ITER)
            print('[*]At iteration %s/%s, loss is %s' % (i, num_iter, loss))


def train(sess, model, trainer, safety_trainer, num_iter):
    model_save_path = os.path.join('../checkpoint', str(TRAIN_ITER))
    summary_save_path = os.path.join('../summary', str(TRAIN_ITER))
    # create folders
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(summary_save_path):
        os.makedirs(summary_save_path)

    # load data for primary policy
    x, y, _, _ = load_data(TRAIN_ITER, display=DISPLAY)

    # load data for training safety policy
    safety_x, pi_label, _, _ = load_data(TRAIN_ITER, display=DISPLAY, safety=True)

    # concate dataset
    safety_x = np.concatenate((safety_x, x), axis=0)
    pi_label = np.concatenate((pi_label, y), axis=0)

    # summary writer
    writer = tf.summary.FileWriter(summary_save_path, graph=sess.graph)

    # primary policy
    train_primary_policy(sess, x, y, writer, num_iter=num_iter, trainer=trainer, model=model)

    # safety policy
    train_safety_policy(sess, safety_x, pi_label, writer, model, num_iter, safety_trainer)


if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess)
        model = NeuralCommander(BATCH_SIZE)
        # trainer for primary policy
        s = 'gradient_norm'
        primary_policy_trainer = optimize_loss(
            model.loss, model.global_pi_setp, learning_rate=0.0005, name='primary_optimizer',
            optimizer='Adam', variables=[v for v in tf.trainable_variables() if 'cnn' in v.name],
            summaries=[s]
        )

        # trainer for safety policy
        safety_policy_trainer = optimize_loss(
            model.safety_loss, model.global_safety_step, learning_rate=0.0009, name='safety_optimizer',
            optimizer='Adam', variables=[v for v in tf.trainable_variables() if 'safety_policy' in v.name]
        )

        # initialize all variables
        if TRAIN_ITER > 0:
            model.restore(sess, TRAIN_ITER-1)
        sess.run(tf.global_variables_initializer())
        train(sess, model, primary_policy_trainer, safety_policy_trainer, NUM_ITERS)
        convert_to_pkl(model, sess, TRAIN_ITER)


