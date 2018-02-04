import tensorflow as tf
import scipy.sparse
from time import time
from DBConnector import DBConnector

# Constant
hidden_unit = 128  # hidden layer units
input_size = 7649
output_size = 7649
lr = 0.0005  # Learning rate

# LSTM parameters
time_step = 8
batch_size = 100
X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])

# Connector instance
inDBConnector = DBConnector()

# Others
length_ts = 12


def get_data():
    # Get game list
    query = 'SELECT DISTINCT gameid FROM ts_train_data'
    result_list = inDBConnector.runQuery(query)
    game_list = list(map(list, zip(*result_list)))[0]

    # Get user list
    query = 'SELECT userid, COUNT(*) AS num FROM raw_train_data GROUP BY userid ORDER BY num DESC LIMIT 2000'
    user_list = inDBConnector.runQuery(query)

    train_x = []
    train_y = []

    for user in user_list:
        query = 'SELECT gameid, ratings FROM ts_train_data WHERE userid = \'%s\'' % user[0]
        rating_list = inDBConnector.runQuery(query)

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

        for i in range(length_ts - time_step):
            train_x.append(rating_mtx[i:time_step + i])
            train_y.append(rating_mtx[i + 1:time_step + i + 1])

    return train_x, train_y


def get_test():
    # Get game list
    query = 'SELECT DISTINCT gameid FROM ts_train_data'
    result_list = inDBConnector.runQuery(query)
    game_list = list(map(list, zip(*result_list)))[0]

    # Get user list
    query = 'SELECT userid, COUNT(*) AS num FROM raw_train_data GROUP BY userid ORDER BY num DESC LIMIT 2000'
    user_list = inDBConnector.runQuery(query)

    test_x = []
    test_y = []
    known = []

    for user in user_list:
        query = 'SELECT gameid, ratings FROM ts_train_data WHERE userid = \'%s\'' % user[0]
        rating_list = inDBConnector.runQuery(query)
        known_game = list(map(list, zip(*rating_list)))[0]
        known.append(known_game)

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
        result_list = inDBConnector.runQuery(query)
        if result_list:
            result_list = list(map(list, zip(*result_list)))[0]
        test_y.append(result_list)

    return game_list, test_x, test_y, known


def lstm(batch):
    # Define Input weights and biases
    w_in = tf.Variable(tf.random_normal([input_size, hidden_unit]))
    b_in = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]))
    # Change tensor to 2-dimension
    # Results are input to hidden layer
    input_ = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input_, w_in) + b_in
    # Change tensor to 3-dimension
    # Results are input to to LSTM cells
    input_rnn = tf.reshape(input_rnn, [-1, time_step, hidden_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    # output_rnn: results of every LSTM cells
    # final_states: result of the last LSTM cell
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)
    # Input to output layer
    output = tf.reshape(output_rnn, [-1, hidden_unit])
    w_out = tf.Variable(tf.random_normal([hidden_unit, output_size]))
    b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


def train():
    train_x, train_y = get_data()
    # Run time
    time_start = time()
    pred, _ = lstm(batch_size)
    # loss function
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    # Save model
    saver = tf.train.Saver(tf.global_variables())
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        # Training times
        for turn in range(200):
            loss_ = None
            for start in range(0, len(train_x) - batch_size, batch_size):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:start + batch_size],
                                                                 Y: train_y[start:start + batch_size]})
            print(loss_)
        saver.save(sess, './models/NoSVD_%d/NoSVD.model' % hidden_unit, global_step=200)
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
        module_file = tf.train.latest_checkpoint('./models/NoSVD_%d/' % hidden_unit)
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
    prediction()
