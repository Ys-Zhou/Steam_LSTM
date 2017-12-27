# -*- coding: utf-8 -*-
from DBConnector import DBConnector

inDBConnector = DBConnector()

k = 12  # 目标列数


def listadd(list_a, list_b):
    return [x + y for x, y in zip(list_a, list_b)]


def listpro(list_, pro):
    return [x * pro for x in list_]


def avg(uname):
    query = ("SELECT gameid, d1, d2, d3, d4, d5, d6 FROM raw_train_data WHERE userid = \'%s\'") % uname
    result_list = inDBConnector.runQuery(query)
    table = list(map(list, zip(*result_list)))
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
        inDBConnector.runInsert(insert, (uname, table[0][i], ratings))


if __name__ == '__main__':
    query_ = 'SELECT DISTINCT userid FROM raw_train_data'
    result_list_ = inDBConnector.runQuery(query_)
    count = 1
    for item in result_list_:
        avg(item[0])
        print(count, item[0])
        count += 1
