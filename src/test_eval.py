import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xml.dom.pulldom import PROCESSING_INSTRUCTION
import model
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import psycopg2
import configparser
import time

def all_plan_costs(net, f_input, f_output, show=True):
    print("All Plan Costs")
    # Load test data.
    with open(f_input, 'r') as f_data:
        data = json.load(f_data)

    # We need to store list of tuples for all the known costs:
    # (plan no., [cost per query], alpha)
    print("Recording known costs...")
    data_len = int(len(data)/model.NUM_PLANS)
    samples = [(f'Plan {i+1}', data[i*data_len:(i+1)*data_len], 0.2) for i in range(model.NUM_PLANS)]

    # Run through all of the features and infer the cost.
    print("Predicting plan to choose...")
    pred = []
    for i in tqdm(range(data_len)):
        x = torch.tensor(data[i]['x']).unsqueeze(dim=1)
        y = torch.argmin(net(x)[0].squeeze(dim=0)).item()
        pred.append(samples[y][1][i])
    samples.append(('Selected', pred, 1.0))

    # Plot the graphs.
    print("Plotting graph...")
    for c in samples:
        costs = np.array([cst['y'] for cst in c[1]])
        costs = costs/np.linalg.norm(costs)
        plt.plot(range(len(costs)), costs, label=c[0], alpha=c[2])
    plt.title('Plan Costs')
    plt.xlabel("Num Queries")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.savefig(os.path.join(f_output, 'all_plans.png'))
    if show: plt.show()
    plt.clf()


def cost_difference(net, f_input, f_output, show=True):
    print("Cost Difference")
    # Load test data.
    with open(f_input, 'r') as f_data:
        data = json.load(f_data)

    # Predict the cost of each sample.
    print("Predicting costs...")
    pred = []
    actual = []
    for sample in tqdm(data):
        actual.append(sample['y'])
        x = torch.tensor(sample['x']).unsqueeze(dim=1)
        y = net(x)[0].squeeze(dim=0)[sample['a']].item()
        pred.append(y)
    
    # Normalize the vectors and compute the delta.
    print("Calculating norm and delta...")
    pred = np.array(pred)
    actual = np.array(actual)
    pred = pred/np.linalg.norm(pred)
    actual = actual/np.linalg.norm(actual)
    delta = np.abs(np.subtract(pred, actual))
    delta = np.array([(i, delta[i]) for i in range(len(delta))])
    delta = sorted(delta, key=lambda x: x[1], reverse=True)
    pred = np.array([pred[int(i[0])] for i in delta])
    actual = np.array([actual[int(i[0])] for i in delta])
    delta = np.array([i[1] for i in delta])

    # Plot the graphs.
    print("Plotting graph...")
    plt.plot(range(len(pred)), pred, label='Predicted')
    plt.plot(range(len(pred)), actual, label='Actual')
    plt.plot(range(len(pred)), delta, label='Delta')
    plt.title('Actual vs Predicted Cost')
    plt.xlabel("Queries")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.savefig(os.path.join(f_output, 'delta.png'))
    if show: plt.show()
    plt.clf()


def query_timer(net, f_input, f_output, show=True):
    print("Query Timer")
    # Load the queries into an array.
    print("Loading SQL...")
    query_directory = os.path.join('queries', 'imdb', 'test')
    subs = os.listdir(query_directory)
    queries = []
    for s_file in subs:
        s_file = os.path.join(query_directory, s_file)
        with open(s_file, 'r', encoding="utf8") as sql_file:
            sql = sql_file.read()
            queries.append(sql)

    # Connect to the database.
    print("Connecting to DB...")
    config = configparser.ConfigParser()
    config.read('server.cfg')
    cfg = config['db']
    conn = psycopg2.connect(
        database=cfg['database'], user=cfg['user'],
        password=cfg['password'], host=cfg['host'], port=int(cfg['port'])
    )
    conn.autocommit = True
    cursor = conn.cursor()
    conn.commit()

    # Execute all queries once to warm up the db.
    print("Warming up db...")
    for sql in tqdm(queries):
        cursor.execute(sql)

    # Run through all of the queries with no hints.
    print("Timing SQL without hints...")
    cursor.execute("set enable_bao to on;")
    #cursor.execute("set bao_host to \"localhost\";")
    cursor.execute("set enable_bao_selection to off;")
    conn.commit()
    no_plan = []
    start = time.time()
    for sql in tqdm(queries):
        cursor.execute(sql)
        no_plan.append(time.time()-start)

    # Run through all of the queries with hints.
    print("Timing SQL with hints...")
    cursor.execute("set enable_bao_selection to on;")
    conn.commit()
    plan = []
    start = time.time()
    for sql in tqdm(queries):
        cursor.execute(sql)
        plan.append(time.time()-start)

    # Build out ranges for the max time.
    max_time = max([sum(no_plan), sum(plan)])
    
    # Plot the graphs.
    print("Plotting graph...")
    plt.plot(range(len(no_plan)), no_plan, label='No Learned Optimizer')
    plt.plot(range(len(no_plan)), plan, label='Learned Optimizer')
    plt.title('Query Exec Time')
    plt.xlabel("Num Queries")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.savefig(os.path.join(f_output, 'query_time.png'))
    if show: plt.show()
    plt.clf()


if __name__ == '__main__':
    # Load the model to evaluate.
    f_net = os.path.join('models', 'Lstm2xNetwork', '5000_5e-05_32_50_', 'model.pt')
    net = model.Lstm2XNetwork(model.NUM_FEATURES, model.NUM_PLANS)
    net.load_state_dict(torch.load(f_net, map_location=torch.device('cpu')))
    net.eval()

    # Run through the evals.
    os.makedirs('metrics', exist_ok=True)
    metrics = 'metrics'
    test_data = os.path.join('data', 'testing.json')
    all_plan_costs(net, test_data, metrics, show=False)
    cost_difference(net, test_data, metrics, show=False)
    query_timer(net, test_data, metrics, show=False)