import os
import random
import psycopg2
import configparser
import time
from tqdm import tqdm
import multiprocessing


def build_query_list():
    queries = []

    query_directory = os.path.join('queries', 'imdb')
    #sub_directories = os.listdir(query_directory)
    sql_files = os.listdir(query_directory)

    for s_file in sql_files:
        s_file = os.path.join(query_directory, s_file)
        with open(s_file, 'r', encoding="utf8") as sql_file:
            sql = sql_file.read()
            queries.append(sql)

    """for dir in sub_directories:
        sub_dir = os.path.join(query_directory, dir)
        sql_files = os.listdir(sub_dir)

        for s_file in sql_files:
            with open(os.path.join(sub_dir, s_file), 'r', encoding="utf8") as sql_file:
                sql = sql_file.read()
                queries.append(sql)"""

    #random.shuffle(queries)
    return queries


def query_worker(query_list, cursor):
    # Run the queries.
    for i in range(len(query_list)):
        sql = query_list[i]
        cursor.execute(sql)
        print(os.getp)


def run_queries(query_list, threads=2, increment=2):
    # Get the database config.
    config = configparser.ConfigParser()
    config.read('server.cfg')
    cfg = config['db']

    # Connect to the database.
    conn = psycopg2.connect(
        database=cfg['database'], user=cfg['user'],
        password=cfg['password'], host=cfg['host'], port=int(cfg['port'])
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # Enable Bao
    cursor = conn.cursor()
    cursor.execute("set enable_bao to on;")
    cursor.execute("set bao_host to \"localhost\";")
    conn.commit()
    #time.sleep(5)

    # Cycle through and run all queries.
    print("Running queries...")
    start = time.time()
    for i in tqdm(range(len(query_list))):
        sql = query_list[i]
        cursor.execute(sql)
    """with multiprocessing.Pool(processes=threads) as pool:
        chunks = [(query_list[i:i+increment], cfg) for i in range(0, 4, increment)]
        result = pool.starmap(query_worker, chunks)"""
    print("Queries complete! Shutting down. Time: %ds" % (time.time()-start))

    # Close the database connection.
    time.sleep(10)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    query_list = build_query_list()
    run_queries(query_list)