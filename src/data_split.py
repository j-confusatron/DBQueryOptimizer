import json
import os
from tqdm import tqdm

def load_and_split(data_file, split=[8,1,1]):
    assert sum(split) > 0, "Split definition must contain positive designations."

    # Read in the raw json.
    with open(data_file, 'r') as fp:
        data_json = json.load(fp)

    # Split the data.
    i_split = 0
    data = [[] for _ in range(len(split))]
    cur = split[0]
    print("Splitting data...")
    for i in tqdm(range(len(data_json))):
        while cur <= 0:
            i_split += 1
            if i_split == len(split):
                i_split = 0
            cur = split[i_split]
        data[i_split].append(data_json[i])
        cur -= 1

    # Return the data.
    return data

if __name__ == '__main__':
    raw_data = os.path.join('data', 'training.json')
    data = load_and_split(raw_data)
    print("Writing split data...")
    for d, n in zip(data, ['tr_data.json', 'val_data.json', 'test_data.json']):
        print(f"{n}: {len(d)}")
        with open(os.path.join('data', n), 'w') as fp:
            json.dump(d, fp)
    print("Split complete.")