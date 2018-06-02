import tensorflow as tf
import os
from dataset import DataSet
from models import Models

# LSTM parameters
input_size = output_size = 7649
time_step = 8
batch_size = 128

# training parameters
learning_rate = 0.0001
training_steps = 300

# other parameters
user_limit = 2000


def train(hidden_unit: int):
    # placeholder and data set
    x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    train_x, train_y = DataSet(time_step, user_limit).get_data()

    # Load model
    prd = Models.lstm(x, input_size, hidden_unit, output_size, time_step, batch_size)

    # error and optimize function
    with tf.name_scope('train'):
        error = tf.reduce_mean(tf.abs(tf.subtract(prd, y)))
        tf.summary.scalar('error', error)
        update_op = tf.train.AdamOptimizer(learning_rate).minimize(error)

    # initialize or reload model
    saver = tf.train.Saver(tf.global_variables())
    # module_file = tf.train.latest_checkpoint()

    # Run session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # Merge summaries
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('models/NoSVD_%d/log' % hidden_unit, sess.graph)

        # initialize or reload model
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)

        # Start learning
        curr_err = 1
        for turn in range(training_steps):
            last_err = curr_err
            feed_dict = {}
            for start in range(0, len(train_x) - batch_size, batch_size):
                feed_dict = {x: train_x[start:start + batch_size], y: train_y[start:start + batch_size]}
                _, curr_err = sess.run([update_op, error], feed_dict=feed_dict)

            # Write summaries
            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=turn)

            print('Step %d: current error = %g (%g)' % (turn, curr_err, curr_err / last_err))

        # Save model
        saver.save(sess, 'models/NoSVD_%d/NoSVD.model' % hidden_unit, global_step=training_steps)


if __name__ == '__main__':
    try:
        train(256)
    finally:
        # os.system('shutdown /s /t 60')
        pass
