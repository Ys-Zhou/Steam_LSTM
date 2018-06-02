import scipy.sparse
from dbconnector import GetCursor


class DataSet:
    def __init__(self, user_limit: int, time_step: int = None):
        with GetCursor() as cur:
            # Get game list
            query = 'SELECT DISTINCT gameid FROM ts_train_data ORDER BY gameid'
            cur.execute(query)
            self.__game_dict = dict((k[0], v) for v, k in enumerate(cur))
            self.__game_count = len(self.__game_dict)

            # Get user list
            query = 'SELECT userid, COUNT(*) AS num FROM raw_train_data GROUP BY userid ORDER BY num DESC LIMIT %d' \
                    % user_limit
            cur.execute(query)
            self.__user_list = cur.fetchall()

            self.__time_step = time_step

    def __get_rt_mtx(self, user: str):
        with GetCursor() as cur:
            # Get ratings
            query = 'SELECT gameid, ratings FROM ts_train_data WHERE userid = \'%s\'' % user
            cur.execute(query)

            # [row, col, val]
            sparse_matrix = [[], [], []]
            for rating in cur:
                ratings = rating[1].split()  # str
                self.__ts_count = len(ratings)
                for i in range(self.__ts_count):
                    if float(ratings[i]) > 0:
                        sparse_matrix[0].append(i)
                        sparse_matrix[1].append(self.__game_dict[rating[0]])
                        sparse_matrix[2].append(float(ratings[i]))

        return sparse_matrix

    # Prepare training data for lstm
    def lstm_train(self):
        train_x = []
        train_y = []

        for user in self.__user_list:
            sparse_matrix = self.__get_rt_mtx(user[0])
            rating_mtx = scipy.sparse.coo_matrix((sparse_matrix[2], sparse_matrix[0:2]),
                                                 shape=(self.__ts_count, self.__game_count)).toarray()

            for i in range(self.__ts_count - self.__time_step):
                train_x.append(rating_mtx[i:self.__time_step + i])
                train_y.append(rating_mtx[i + 1:self.__time_step + i + 1])

        return train_x, train_y

    # Prepare testing data for lstm
    def lstm_test(self):
        test_x = []
        test_y = []
        known = []

        for user in self.__user_list:
            sparse_matrix = self.__get_rt_mtx(user[0])
            rating_mtx = scipy.sparse.coo_matrix((sparse_matrix[2], sparse_matrix[0:2]),
                                                 shape=(self.__ts_count, self.__game_count)).toarray()

            test_x.append(rating_mtx[self.__ts_count - self.__time_step:self.__ts_count])

            query = 'SELECT gameid FROM date170709 WHERE userid = \'%s\'' % user[0]
            with GetCursor() as cur:
                cur.execute(query)
                test_list = []
                for row in cur:
                    if row[0] in self.__game_dict:
                        test_list.append(self.__game_dict[row[0]])
            test_y.append(test_list)

            known.append(sparse_matrix[1])

        return test_x, test_y, known

    # Prepare testing data for codec
    def codec_train(self):
        train = []

        for user in self.__user_list:
            sparse_matrix = self.__get_rt_mtx(user[0])
            rating_mtx = scipy.sparse.coo_matrix((sparse_matrix[2], sparse_matrix[0:2]),
                                                 shape=(self.__ts_count, self.__game_count)).toarray()
            train.append(rating_mtx)

        return train
