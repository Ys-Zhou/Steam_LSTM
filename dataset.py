import scipy.sparse
from dbconnector import GetCursor


class DataSet:
    def __init__(self, user_limit: int, time_step: int = None):
        with GetCursor() as cur:
            # Get game list
            query = 'SELECT DISTINCT gameid FROM ts_train_data ORDER BY gameid'
            cur.execute(query)
            self.game_tpl = list(zip(*cur.fetchall()))[0]

            self.__game_dict = dict((k, v) for v, k in enumerate(self.game_tpl))
            self.__game_count = len(self.game_tpl)

            # Get user list
            query = 'SELECT userid, COUNT(*) AS num FROM raw_train_data GROUP BY userid ORDER BY num DESC LIMIT %d' \
                    % user_limit
            cur.execute(query)
            self.__user_tpl = list(zip(*cur.fetchall()))[0]
            self.__user_sparse_matrix = None

            self.__time_step = time_step

    def __require_usm(self):
        if not self.__user_sparse_matrix:
            self.__user_sparse_matrix = dict()

            with GetCursor() as cur:
                for user in self.__user_tpl:
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

                    self.__user_sparse_matrix.setdefault(user, sparse_matrix)

    # Prepare training data for lstm
    # return: shape=[?, time_step, item_size]
    def lstm_train(self) -> (list, list):
        self.__require_usm()

        train_x = []
        train_y = []

        for user in self.__user_tpl:
            sparse_matrix = self.__user_sparse_matrix[user]
            rating_mtx = scipy.sparse.coo_matrix((sparse_matrix[2], sparse_matrix[0:2]),
                                                 shape=(self.__ts_count, self.__game_count)).toarray()

            for i in range(self.__ts_count - self.__time_step):
                train_x.append(rating_mtx[i:self.__time_step + i])
                train_y.append(rating_mtx[i + 1:self.__time_step + i + 1])

        return train_x, train_y

    # Prepare testing data for lstm
    # return: shape=[user_size, time_step, item_size]
    def lstm_test(self) -> list:
        self.__require_usm()

        test_x = []

        for user in self.__user_tpl:
            sparse_matrix = self.__user_sparse_matrix[user]
            rating_mtx = scipy.sparse.coo_matrix((sparse_matrix[2], sparse_matrix[0:2]),
                                                 shape=(self.__ts_count, self.__game_count)).toarray()

            test_x.append(rating_mtx[self.__ts_count - self.__time_step:self.__ts_count])

        return test_x

    # return: shape=[user_size, ?], shape 0 contains gameid
    def correct_data(self) -> (list, list):
        test_y = []
        known = []

        with GetCursor() as cur:
            for user in self.__user_tpl:
                test_list = []
                query = 'SELECT gameid FROM date170709 WHERE userid = \'%s\'' % user
                cur.execute(query)
                for row in cur:
                    if row[0] in self.__game_dict:
                        test_list.append(row[0])
                test_y.append(test_list)

                query = 'SELECT gameid FROM ts_train_data WHERE userid = \'%s\'' % user
                cur.execute(query)
                known.append(list(zip(*cur.fetchall()))[0])

        return test_y, known

    # return: shape=[user_size, ts_count, item_size]
    def all_data(self) -> (list, list):
        self.__require_usm()

        data = []
        sign = []

        for user in self.__user_tpl:
            sparse_matrix = self.__user_sparse_matrix[user]
            rating_mtx = scipy.sparse.coo_matrix((sparse_matrix[2], sparse_matrix[0:2]),
                                                 shape=(self.__ts_count, self.__game_count)).toarray()
            sign_mtx = scipy.sparse.coo_matrix(([1] * len(sparse_matrix[2]), sparse_matrix[0:2]),
                                               shape=(self.__ts_count, self.__game_count)).toarray()
            data.append(rating_mtx)
            sign.append(sign_mtx)

        return data, sign

    # return: shape=[user_size, item_size]
    def global_data(self) -> (list, list):
        data = []
        sign = []

        with GetCursor() as cur:
            for user in self.__user_tpl:
                # Get global rating
                query = 'SELECT gameid, rating FROM global_train_data WHERE userid = \'%s\'' % user
                cur.execute(query)

                rating_list = [0.0] * self.__game_count
                sign_list = [0] * self.__game_count
                for row in cur:
                    rating_list[self.__game_dict[row[0]]] = float(row[1])
                    sign_list[self.__game_dict[row[0]]] = 1
                data.append(rating_list)
                sign.append(sign_list)

        return data, sign
