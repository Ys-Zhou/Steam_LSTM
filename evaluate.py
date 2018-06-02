import tensorflow as tf
from dataset import DataSet
from models import Models

# LSTM parameters
input_size = output_size = 7649
time_step = 8

# other parameters
user_limit = 2000


# By Recall(Top-50)
def evaluate(hidden_unit: int):
    x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    test_x, test_y, known = DataSet(time_step, user_limit).get_test()

    prd = Models.lstm(x, input_size, hidden_unit, output_size, time_step, batch_size=1)

    saver = tf.train.Saver(tf.global_variables())

    hits = 0
    crrs = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
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
    evaluate(256)
