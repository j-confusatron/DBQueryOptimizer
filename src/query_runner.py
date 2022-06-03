import os
import random
import psycopg2
import configparser
import time
from tqdm import tqdm
import argparse


def build_query_list(dataset, shuffle=False):
    queries = []

    # Get the root query directory.
    if dataset == 'imdb-test':
        query_directory = os.path.join('queries', 'imdb', 'test')
    elif dataset == 'imdb-train':
        query_directory = os.path.join('queries', 'imdb', 'train')
    else:
        query_directory = os.path.join('queries', 'so_queries')
    subs = os.listdir(query_directory)

    # IMDB
    if dataset == 'imdb-test' or dataset == 'imdb-train':
        for s_file in subs:
            s_file = os.path.join(query_directory, s_file)
            with open(s_file, 'r', encoding="utf8") as sql_file:
                sql = sql_file.read()
                queries.append(sql)

    # SO
    elif dataset == 'so':
        for dir in subs:
            sub_dir = os.path.join(query_directory, dir)
            sql_files = os.listdir(sub_dir)

            for s_file in sql_files:
                with open(os.path.join(sub_dir, s_file), 'r', encoding="utf8") as sql_file:
                    lines = sql_file.readlines()
                    sql = ''
                    for l in lines:
                        if l != '\n' and '--' not in l:
                            sql += f' {l}'
                    #sql = sql_file.read()
                    sql = sql.strip().replace('\n', ' ')
                    queries.append(sql)
                    #queries.append(sql.strip().replace('\n', ' '))

    # Return the queries.
    if shuffle:
        random.shuffle(queries)
    return queries

def connect(cfg):
    conn = psycopg2.connect(
        database=cfg['database'], user=cfg['user'],
        password=cfg['password'], host=cfg['host'], port=int(cfg['port'])
    )
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute("set enable_bao to on;")
    #cursor.execute("set bao_host to \"localhost\";")
    conn.commit()
    return conn, cursor

def run_queries(query_list, threads=2, increment=2):
    # Get the database config.
    config = configparser.ConfigParser()
    config.read('server.cfg')
    cfg = config['db']

    # Connect to the database.
    conn, cursor = connect(cfg)

    # Cycle through and run all queries.
    print("Running queries...")
    start = time.time()
    for i in tqdm(range(len(query_list))):
        sql = query_list[i]
        try:
            cursor.execute(sql)
        except:
            print(sql)
            conn, cursor = connect(cfg)
    print("Queries complete! Shutting down. Time: %ds" % (time.time()-start))

    # Close the database connection.
    conn.commit()
    time.sleep(5)
    cursor.close()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False, default='so', choices=['imdb-train', 'imdb-test', 'so'])
    args = parser.parse_args()
    query_list = build_query_list(args.dataset, shuffle=True)
    run_queries(query_list)