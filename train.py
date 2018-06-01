import tensorflow as tf
from dataset import DataSet
from models import Models

# LSTM parameters
input_size = output_size = 7649
hidden_unit = 16
time_step = 8
batch_size = 128

# training parameters
learning_rate = 0.0005
max_ls = 200  # Max learning step
end_cnd = 0.999  # End condition (RMSE)

# other parameters
length_ts = 12  # Time series length
user_limit = 2000


def train():
    # placeholder and data set
    x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    train_x, train_y = DataSet(time_step, user_limit).get_data()
    # Load model
    prd = Models.lstm(input_size, hidden_unit, output_size, time_step, batch_size)
    # loss and evaluation function
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.reshape(prd, [-1]) - tf.reshape(y, [-1])))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # initialize or reload model
    saver = tf.train.Saver(tf.global_variables())
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        # Merge summaries
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('models/NoSVD_%d/log' % hidden_unit, sess.graph)
        # initialize or reload model
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        last_loss = 9999
        crrn_loss = 999
        turn = 0
        while turn < max_ls and crrn_loss / last_loss < end_cnd:
            last_loss = crrn_loss
            turn += 1
            feed_dict = {}
            for start in range(0, len(train_x) - batch_size, batch_size):
                feed_dict = {x: train_x[start:start + batch_size], y: train_y[start:start + batch_size]}
                _, crrn_loss = sess.run([train_op, loss], feed_dict=feed_dict)
            # Write summaries
            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=turn)
            print('Step %d: current loss = %g (%g)' % (turn, crrn_loss, crrn_loss / last_loss))
        saver.save(sess, 'models/NoSVD_%d/NoSVD.model' % hidden_unit, global_step=turn)


# By Recall(Top-50)
def prediction():
    hits = 0
    crrs = 0
    x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    test_x, test_y, known = DataSet(time_step, user_limit).get_test()
    prd = Models.lstm(input_size, hidden_unit, output_size, time_step, batch_size=1)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('models/NoSVD_%d/' % hidden_unit)
        saver.restore(sess, module_file)
        for i in range(len(test_x)):
            next_ = sess.run(prd, feed_dict={x: [test_x[i]]})
            priority = [[i, p] for p, i in enumerate(next_[-1])]
            priority.sort(reverse=True)
            priority = list(map(list, zip(*priority)))[1]
            rec = [g for g in priority if g not in known[i]][:50]
            crr = [h for h in test_y[i] if h not in known[i]]
            hits += len(set(rec).intersection(set(crr)))
            crrs += len(crr)
            print('[hits, corrs] = [%d, %d]' % (hits, crrs))
    print('recall = %g' % (hits / crrs))


if __name__ == '__main__':
    # train()
    prediction()
    pass
