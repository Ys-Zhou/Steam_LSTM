from mysql.connector.pooling import MySQLConnectionPool
from mysql.connector.cursor import MySQLCursor


# Singleton pattern decorator
def singleton(cls: type, *args, **kw):
    instance = {}

    def _singleton():
        if cls not in instance:
            instance[cls] = cls(*args, **kw)
        return instance[cls]

    return _singleton


# Get database connection pool
@singleton
class _DBConnector:

    def __init__(self):
        cnf = {
            'user': 'root',
            'password': 'zhou',
            'host': 'localhost',
            'port': 3306,
            'database': 'steam',
            'charset': 'utf8mb4',
            'pool_name': 'my_pool',
            'pool_size': 5
        }
        self.cnx_pool = MySQLConnectionPool(**cnf)


# Create a cursor from a connection
class GetCursor:

    def __enter__(self) -> MySQLCursor:
        self.__cnx = _DBConnector().cnx_pool.get_connection()
        self.__cur = self.__cnx.cursor()
        return self.__cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is None:
            self.__cnx.commit()
        else:
            self.__cnx.rollback()
        self.__cur.close()
        self.__cnx.close()
