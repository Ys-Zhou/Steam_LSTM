import tensorflow as tf
from dataset import DataSet
from models import Models

# LSTM parameters
input_size = output_size = 7649
time_step = 8
batch_size = 128

# training parameters
learning_rate = 0.0001
max_ls = 200  # max learning step

# other parameters
length_ts = 12  # Time series length
user_limit = 2000


def train(hidden_unit: int):
    # placeholder and data set
    x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    train_x, train_y = DataSet(time_step, user_limit).get_data()

    # Load model
    prd = Models.lstm(x, input_size, hidden_unit, output_size, time_step, batch_size)

    # error and evaluation function
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
        for turn in range(max_ls):
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
        saver.save(sess, 'models/NoSVD_%d/NoSVD.model' % hidden_unit, global_step=max_ls)


# By Recall(Top-50)
def prediction(hidden_unit: int):
    x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    test_x, test_y, known = DataSet(time_step, user_limit).get_test()

    prd = Models.lstm(x, input_size, hidden_unit, output_size, time_step, batch_size=1)

    saver = tf.train.Saver(tf.global_variables())

    hits = 0
    crrs = 0
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('models/NoSVD_%d/' % hidden_unit)
        saver.restore(sess, module_file)

        for i in range(len(test_x)):
            next_ = sess.run(prd, feed_dict={x: [test_x[i]]})

            priority = [[i, p] for p, i in enumerate(next_[0][-1])]
            priority.sort(reverse=True)
            priority = list(map(list, zip(*priority)))[1]

            rec = [g for g in priority if g not in known[i]][:50]
            crr = [h for h in test_y[i] if h not in known[i]]

            hits += len(set(rec).intersection(set(crr)))
            crrs += len(crr)

            print('[hits, corrs] = [%d, %d]' % (hits, crrs))

    print('recall = %g' % (hits / crrs))


if __name__ == '__main__':
    train(128)
    # prediction(128)
    pass
