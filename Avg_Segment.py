# -*- coding: utf-8 -*-
from DBConnector import GetCursor

k = 12  # 目标列数


def listadd(list_a, list_b):
    return [x + y for x, y in zip(list_a, list_b)]


def listpro(list_, pro):
    return [x * pro for x in list_]


def avg(uname):
    with GetCursor() as cur:
        query = 'SELECT gameid, d1, d2, d3, d4, d5, d6 FROM raw_train_data WHERE userid = \'%s\'' % uname
        cur.execute(query)
        table = list(map(list, zip(*cur.fetchall())))
        lent = len(table[0])

        all_ = 0
        for line in table[1:]:
            all_ += sum(line)
        group_time = float(all_) / k

        remind = [0] * lent
        aim = []
        for line in table[1:]:
            new = line
            while sum(remind) + sum(new) > group_time * 0.99999:
                proportion = (group_time - sum(remind)) / sum(new)
                res_list = listadd(remind, listpro(new, proportion))
                rate_list = listpro(res_list, 1 / group_time)
                aim.append(rate_list)
                remind = [0] * lent
                new = listpro(new, 1 - proportion)
            remind = listadd(remind, new)
        rating_list = list(map(list, zip(*aim)))

        for i in range(lent):
            ratings = ' '.join(map(lambda x: str(round(x, 3)), rating_list[i]))
            insert = 'INSERT INTO ts_train_data VALUES (%s, %s, %s)'
            cur.execute(insert, (uname, table[0][i], ratings))


if __name__ == '__main__':
    with GetCursor() as cur_:
        query_ = 'SELECT DISTINCT userid FROM raw_train_data'
        cur_.execute(query_)
        count = 1
        for row in cur_:
            avg(row[0])
            print(count, row[0])
            count += 1
