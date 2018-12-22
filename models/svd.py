import tensorflow as tf
from dataset import DataSet
from dbwriter import DbWriter
import time


class SVD:

    def __init__(self, item_size: int, user_size: int, hidden_size: int):
        self.user_size = user_size
        self.hidden_size = hidden_size

        self.y = tf.placeholder(tf.float32, shape=[user_size, item_size], name='train_data')
        self.sign = tf.placeholder(tf.float32, shape=[user_size, item_size], name='sign_data')

        with tf.name_scope('item_matrix'):
            item_mtx = tf.Variable(tf.random_normal([hidden_size, item_size]))
        with tf.name_scope('user_matrix'):
            user_mtx = tf.Variable(tf.random_normal([user_size, hidden_size]))
        with tf.name_scope('like_svd'):
            self.output = user_mtx @ item_mtx

    def train(self, start_learning_rate: float, training_steps: int, decay_rate: float):
        # data set
        train_data, train_sign = DataSet(self.user_size).global_data()

        # error and optimize function
        with tf.name_scope('train'):
            error = tf.reduce_mean(tf.abs(self.output * self.sign - self.y))
            tf.summary.scalar('error', error)
            # Dynamic learning rate
            global_step = tf.placeholder(tf.int16, name='global_step')
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, training_steps, decay_rate)
            tf.summary.scalar('learning_rate', learning_rate)
            update_op = tf.train.AdamOptimizer(learning_rate).minimize(error)

        # Run session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Merge summaries
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('saved_models/SVD_%d/log' % self.hidden_size, sess.graph)

            # initialize model
            sess.run(tf.global_variables_initializer())

            # Start learning
            for turn in range(training_steps):
                feed_dict = {self.y: train_data, self.sign: train_sign, global_step: turn}
                _, curr_err, curr_lr, summary = sess.run([update_op, error, learning_rate, merged], feed_dict=feed_dict)

                # Write summaries
                summary_writer.add_summary(summary, global_step=turn)

                print('Step %d: error = %g, learning rate = %g' % (turn, curr_err, curr_lr))

            # Save model
            saver = tf.train.Saver()
            saver.save(sess, 'saved_models/SVD_%d/model' % self.hidden_size, global_step=training_steps)

    def evaluate(self):
        dataset = DataSet(self.user_size)
        test_y, known = dataset.correct_data()

        hits = 0
        crrs = 0

        rec_games = []
        hit_games = []

        start = time.time()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            module_file = tf.train.latest_checkpoint('saved_models/SVD_%d/' % self.hidden_size)
            saver = tf.train.Saver()
            saver.restore(sess, module_file)

            prd = sess.run(self.output)

            for user in range(len(prd)):
                priority = list(zip(prd[user], dataset.game_tpl))
                priority.sort(reverse=True)
                priority = list(zip(*priority))[1]

                rec = [g for g in priority if g not in known[user]][:50]
                crr = [h for h in test_y[user] if h not in known[user]]
                hit = list(set(rec).intersection(set(crr)))

                hit_games += hit
                rec_games += rec

                hits += len(hit)
                crrs += len(crr)

                print('[hits, corrs] : [%d, %d]' % (hits, crrs))

        end = time.time()
        print(end - start)

        DbWriter.write(hit_games, 'game_count_svd_hit')
        DbWriter.write(rec_games, 'game_count_svd_rec')

        print('recall = %g' % (hits / crrs))


if __name__ == '__main__':
    model = SVD(item_size=7649, user_size=2000, hidden_size=256)
    # model.train(start_learning_rate=0.05, decay_rate=0.002, training_steps=1000)
    # model.evaluate()
