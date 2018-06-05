from dbconnector import GetCursor
from dataset import DataSet
from dbwriter import DbWriter
import time

cor, _ = DataSet(1000).correct_data()

with GetCursor() as cur:
    with open('evaluate_template.sql', 'r', encoding='utf8') as f:
        query_tmp = f.read()

    query = 'SELECT userid, COUNT(*) AS num FROM raw_train_data GROUP BY userid ORDER BY num DESC LIMIT 1000'
    cur.execute(query)
    users = cur.fetchall()

    rec_games = []
    hit_games = []

    turns = 0
    start = time.time()

    for u in range(len(users)):
        turns += 1
        cur.execute(query_tmp % (users[u][0], users[u][0]))
        for row in cur:
            rec_games.append(row[0])
            if row[0] in cor[u]:
                hit_games.append(row[0])
        print('turn: %d, hit_sum=%d' % (turns, len(hit_games)))

    end = time.time()
    print(end - start)

    DbWriter.write(rec_games, 'game_count_cf_rec')
    DbWriter.write(hit_games, 'game_count_cf_hit')
