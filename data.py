import scipy.sparse
from DBConnector import GetCursor


class Data:
    def __init__(self, time_step: int, user_limit: int):
        with GetCursor() as cur:
            # Get game list
            query = 'SELECT DISTINCT gameid FROM ts_train_data ORDER BY gameid'
            cur.execute(query)
            self.game_dict = dict((k[0], v) for v, k in enumerate(cur))
            self.game_count = len(self.game_dict)

            # Get user list
            query = 'SELECT userid, COUNT(*) AS num FROM raw_train_data GROUP BY userid ORDER BY num DESC LIMIT %d' \
                    % user_limit
            cur.execute(query)
            self.user_list = cur.fetchall()

            self.time_step = time_step

    # Prepare training data
    def get_data(self):
        with GetCursor() as cur:

            ts_count = None
            train_x = []
            train_y = []

            for user in self.user_list:
                query = 'SELECT gameid, ratings FROM ts_train_data WHERE userid = \'%s\'' % user[0]
                cur.execute(query)

                # [row, col, val]
                sparse_matrix = [[], [], []]
                for rating in cur:
                    ratings = rating[1].split()  # str
                    ts_count = len(ratings)
                    for i in range(ts_count):
                        if float(ratings[i]) > 0:
                            sparse_matrix[0].append(i)
                            sparse_matrix[1].append(self.game_dict[rating[0]])
                            sparse_matrix[2].append(float(ratings[i]))

                # Convert to dense matrix
                rating_mtx = scipy.sparse.coo_matrix((sparse_matrix[2], sparse_matrix[0:2]),
                                                     shape=(ts_count, self.game_count)).toarray()

                for i in range(ts_count - self.time_step):
                    train_x.append(rating_mtx[i:self.time_step + i])
                    train_y.append(rating_mtx[i + 1:self.time_step + i + 1])

        return ts_count, self.game_count, train_x, train_y

    # Prepare testing data
    def get_test(self):
        with GetCursor() as cur:

            ts_count = None
            test_x = []
            test_y = []
            known_list = []

            for user in self.user_list:
                query = 'SELECT gameid, ratings FROM ts_train_data WHERE userid = \'%s\'' % user[0]
                cur.execute(query)

                # [row, col, val]
                sparse_matrix = [[], [], []]
                for rating in cur:
                    ratings = rating[1].split()  # str
                    ts_count = len(ratings)
                    for i in range(ts_count):
                        if float(ratings[i]) > 0:
                            sparse_matrix[0].append(i)
                            sparse_matrix[1].append(self.game_dict[rating[0]])
                            sparse_matrix[2].append(float(ratings[i]))

                # Known games for the user
                known_list.append(sparse_matrix[1])

                # Convert to dense matrix
                rating_mtx = scipy.sparse.coo_matrix((sparse_matrix[2], sparse_matrix[0:2]),
                                                     shape=(ts_count, self.game_count)).toarray()
                test_x.append(rating_mtx[ts_count - self.time_step:ts_count])

                query = 'SELECT gameid FROM date170709 WHERE userid = \'%s\'' % user[0]
                cur.execute(query)
                test_list = cur.fetchall()
                if test_list:
                    test_list = list(map(list, zip(*test_list)))[0]
                test_y.append(test_list)

        return ts_count, self.game_count, test_x, test_y, known_list