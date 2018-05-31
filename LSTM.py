import tensorflow as tf
import scipy.sparse
from time import time
from dataset import DataSet
from DBConnector import GetCursor

# LSTM constants
hidden_unit = 16
input_size = output_size = 7649

# Training parameters
lr = 0.0005  # Learning rate
max_ls = 200  # Max learning step
end_cnd = 0.999  # End condition (RMSE)
time_step = 8
batch_size = 128
X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])

# Other parameters
length_ts = 12  # Time series length
user_limit = 2000


def lstm(batch: int):
    with tf.name_scope('input_layer'):
        with tf.name_scope('input_weights'):
            w_in = tf.Variable(tf.random_normal([input_size, hidden_unit]))
            tf.summary.histogram('input_weights', w_in)
        with tf.name_scope('input_biases'):
            b_in = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]))
            tf.summary.histogram('input_biases', b_in)
        with tf.name_scope('input'):
            # Reshape tensor to 2-dimension for matrix multiplication
            # (b, s, in) => (b * s, in)
            input_x = tf.reshape(X, [-1, input_size])
            # Linear map
            # (b * s, in) => (b * s, h)
            input_rnn = tf.matmul(input_x, w_in) + b_in
            # Reshape tensor back to 3-dimension
            # (b * s, h) => (b, s, h)
            input_rnn = tf.reshape(input_rnn, [-1, time_step, hidden_unit])
    with tf.name_scope('lstm_cell'):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit)
        init_state = cell.zero_state(batch, dtype=tf.float32)
        # output_rnn: results of every LSTM cells
        # states: states in the last LSTM cell
        output_rnn, states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    with tf.name_scope('output_layer'):
        with tf.name_scope('output_weights'):
            w_out = tf.Variable(tf.random_normal([hidden_unit, output_size]))
            tf.summary.histogram('output_weights', w_out)
        with tf.name_scope('output_biases'):
            b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
            tf.summary.histogram('output_biases', b_out)
        with tf.name_scope('output'):
            # (b, s, h) => (b * s, h)
            output_rnn = tf.reshape(output_rnn, [-1, hidden_unit])
            # (b * s, h) => (b * s, out)
            output_y = tf.matmul(output_rnn, w_out) + b_out
    return output_y, states


def train():
    train_x, train_y = DataSet(time_step, user_limit).get_data()
    # Run time
    time_start = time()
    pred, _ = lstm(batch_size)
    # loss and evaluation function
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
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
        curr_loss = 999
        turn = 0
        while turn < max_ls and curr_loss / last_loss < end_cnd:
            last_loss = curr_loss
            turn += 1
            feed_dict = {}
            for start in range(0, len(train_x) - batch_size, batch_size):
                feed_dict = {X: train_x[start:start + batch_size], Y: train_y[start:start + batch_size]}
                _, curr_loss = sess.run([train_op, loss], feed_dict=feed_dict)
            # Write summaries
            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=turn)
            print('Step %d: current loss = %g (%g)' % (turn, curr_loss, curr_loss / last_loss))
        saver.save(sess, 'models/NoSVD_%d/NoSVD.model' % hidden_unit, global_step=turn)
        time_stop = time()
        print(time_stop - time_start)


# By Recall(Top-50)
def prediction():
    hits = 0
    corrs = 0
    test_x, test_y, known = DataSet(time_step, user_limit).get_test()
    pred, _ = lstm(1)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('models/NoSVD_%d/' % hidden_unit)
        saver.restore(sess, module_file)
        for i in range(len(test_x)):
            next_ = sess.run(pred, feed_dict={X: [test_x[i]]})
            priority = [[i, p] for p, i in enumerate(next_[-1])]
            priority.sort(reverse=True)
            priority = list(map(list, zip(*priority)))[1]
            rec = [g for g in priority if g not in known[i]][:50]
            corr = [h for h in test_y[i] if h not in known[i]]
            hits += len(set(rec).intersection(set(corr)))
            corrs += len(corr)
            print('[hits, corrs] = [%d, %d]' % (hits, corrs))
    print('recall = %g' % (hits / corrs))


if __name__ == '__main__':
    # train()
    prediction()
    pass
