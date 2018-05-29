import tensorflow as tf
import scipy.sparse
from time import time
from DBConnector import GetCursor

# LSTM constants
hidden_unit = 256
input_size = 7649
output_size = 7649

# Training parameters
lr = 0.0005  # Learning rate
ls = 100  # Learning step
time_step = 8
batch_size = 100
X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])

# Other parameters
length_ts = 12  # Time series length
user_limit = 2000


def get_data():
    with GetCursor() as cur:
        # Get game list
        query = 'SELECT DISTINCT gameid FROM ts_train_data'
        cur.execute(query)
        game_list = list(map(list, zip(*cur.fetchall())))[0]

        # Get user list
        query = 'SELECT userid, COUNT(*) AS num FROM raw_train_data GROUP BY userid ORDER BY num DESC LIMIT %d' \
                % user_limit
        cur.execute(query)
        user_list = cur.fetchall()

        # Prepare training data
        train_x = []
        train_y = []
        for user in user_list:
            query = 'SELECT gameid, ratings FROM ts_train_data WHERE userid = \'%s\'' % user[0]
            cur.execute(query)

            row = []
            col = []
            data = []
            for rating in cur:
                game_index = game_list.index(rating[0])
                ratings = rating[1].split()  # str
                for ts_index in range(length_ts):
                    if float(ratings[ts_index]) > 0:
                        row.append(ts_index)
                        col.append(game_index)
                        data.append(float(ratings[ts_index]))
            rating_mtx = scipy.sparse.coo_matrix((data, (row, col)), shape=(length_ts, input_size)).toarray()

            for i in range(length_ts - time_step):
                train_x.append(rating_mtx[i:time_step + i])
                train_y.append(rating_mtx[i + 1:time_step + i + 1])

    return train_x, train_y


def get_test():
    with GetCursor() as cur:
        # Get game list
        query = 'SELECT DISTINCT gameid FROM ts_train_data'
        cur.execute(query)
        game_list = list(map(list, zip(*cur.fetchall())))[0]

        # Get user list
        query = 'SELECT userid, COUNT(*) AS num FROM raw_train_data GROUP BY userid ORDER BY num DESC LIMIT %d' \
                % user_limit
        cur.execute(query)
        user_list = cur.fetchall()

        # Prepare testing data
        test_x = []
        test_y = []
        known = []
        for user in user_list:
            query = 'SELECT gameid, ratings FROM ts_train_data WHERE userid = \'%s\'' % user[0]
            cur.execute(query)
            rating_list = cur.fetchall()
            known_games = list(map(list, zip(*rating_list)))[0]
            known.append(known_games)

            row = []
            col = []
            data = []
            for rating in rating_list:
                game_index = game_list.index(rating[0])
                ratings = rating[1].split()  # str
                for ts_index in range(length_ts):
                    if float(ratings[ts_index]) > 0:
                        row.append(ts_index)
                        col.append(game_index)
                        data.append(float(ratings[ts_index]))
            rating_mtx = scipy.sparse.coo_matrix((data, (row, col)), shape=(length_ts, input_size)).toarray()
            test_x.append(rating_mtx[length_ts - time_step:length_ts])

            query = 'SELECT gameid FROM date170709 WHERE userid = \'%s\'' % user[0]
            cur.execute(query)
            test_list = cur.fetchall()
            if test_list:
                test_list = list(map(list, zip(*test_list)))[0]
            test_y.append(test_list)

    return game_list, test_x, test_y, known


# Summery variables
def variable_summaries(var):
    with tf.name_scope('summaries'):
        # Record mean (avg)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # Record stddev
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        # Record max and min
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # Record distribution
        tf.summary.histogram('histogram', var)


def lstm(batch: int):
    with tf.name_scope('input_layer'):
        with tf.name_scope('input_weights'):
            w_in = tf.Variable(tf.random_normal([input_size, hidden_unit]))
        with tf.name_scope('input_biases'):
            b_in = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]))
        with tf.name_scope('input'):
            # Change tensor to 2-dimension
            # Results are input to hidden layer
            input_ = tf.reshape(X, [-1, input_size])
            input_rnn = tf.matmul(input_, w_in) + b_in
            # Change tensor to 3-dimension
            # Results are input to to LSTM cells
            input_rnn = tf.reshape(input_rnn, [-1, time_step, hidden_unit])
    with tf.name_scope('cell'):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit)
        init_state = cell.zero_state(batch, dtype=tf.float32)
        # output_rnn: results of every LSTM cells
        # states: result of the last LSTM cell
        output_rnn, states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        # Input to output layer
        output = tf.reshape(output_rnn, [-1, hidden_unit])
    with tf.name_scope('output_layer'):
        with tf.name_scope('output_weights'):
            w_out = tf.Variable(tf.random_normal([hidden_unit, output_size]))
            variable_summaries(w_out)
        with tf.name_scope('output_biases'):
            b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
            variable_summaries(b_out)
        with tf.name_scope('output'):
            pred = tf.matmul(output, w_out) + b_out
            variable_summaries(pred)
    return pred, states


def train():
    train_x, train_y = get_data()
    # Run time
    time_start = time()
    pred, states = lstm(batch_size)
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
        # Training times
        for turn in range(ls):
            loss_ = None
            feed_dict = {}
            for start in range(0, len(train_x) - batch_size, batch_size):
                feed_dict = {X: train_x[start:start + batch_size], Y: train_y[start:start + batch_size]}
                states, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
            # Write summaries
            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=turn)
            print('Step %d: loss = %f' % (turn, loss_))
        saver.save(sess, 'models/NoSVD_%d/NoSVD.model' % hidden_unit, global_step=ls)
        time_stop = time()
        print(time_stop - time_start)


# By Recall(Top-50)
def prediction():
    hits = 0
    corrs = 0
    game_list, test_x, test_y, known = get_test()
    pred, _ = lstm(1)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint('models/NoSVD_%d/' % hidden_unit)
        saver.restore(sess, module_file)
        for i in range(len(test_x)):
            next_ = sess.run(pred, feed_dict={X: [test_x[i]]})
            priority = list(map(list, zip(*[next_[-1], game_list])))
            priority.sort(reverse=True)
            priority = list(map(list, zip(*priority)))[1]
            rec = [g for g in priority if g not in known[i]][:50]
            corr = [h for h in test_y[i] if h not in known[i]]
            hits += len(set(rec).intersection(set(corr)))
            corrs += len(corr)
            print(hits, corrs)


if __name__ == '__main__':
    # train()
    # prediction()
    pass
