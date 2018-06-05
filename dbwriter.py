from collections import Counter
from dbconnector import GetCursor


class DbWriter:
    @staticmethod
    def write(games: list, db: str):
        with GetCursor() as cur:
            data = Counter(games).items()
            query = 'INSERT INTO ' + db + ' (gameid, cnt) VALUES (%s, %s)'
            cur.executemany(query, data)
