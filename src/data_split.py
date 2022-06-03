import json
import os
from tqdm import tqdm
import argparse

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
    # Get the cmd args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--splits', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=str, nargs='+', required=True)
    args = parser.parse_args()
    input = args.input
    splits = args.splits
    file_names = args.output

    # Load and split the input file.
    raw_data = os.path.join('data', input)
    data = load_and_split(raw_data, split=splits)

    # Write the split data to output.
    print("Writing split data...")
    for d, n in zip(data, file_names):
        print(f"{n}: {len(d)}")
        with open(os.path.join('data', n), 'w') as fp:
            json.dump(d, fp)
    print("Split complete.")