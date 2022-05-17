import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import itertools
import model
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SqlDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path, 'r') as fp:
            data = json.load(fp)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        t = (torch.tensor(rec['x']), torch.tensor(rec['a']), torch.tensor(rec['y']))
        return t

def collate_fn_pad(list_pairs_seq_target):
    seqs = [seq for seq, a, target in list_pairs_seq_target]
    actions = [a for seq, a, target in list_pairs_seq_target]
    targets = [target for seq, a, target in list_pairs_seq_target]
    seqs_padded_batched = pad_sequence(seqs)   # will pad at beginning of sequences
    actions_batched = torch.stack(actions)
    targets_batched = torch.stack(targets)
    assert seqs_padded_batched.shape[1] == len(actions_batched) == len(targets_batched)
    return seqs_padded_batched, actions_batched, targets_batched

def moving_average(numbers, window_size=10):
    numbers_series = pd.Series(numbers)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    without_nans = moving_averages_list[window_size - 1:]
    return without_nans

def viz_loss(train_loss, val_loss, viz_filepath=None, show_loss=False):
    epochs = np.arange(1, len(train_loss)+1, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validate Loss')
    plt.title('Loss (Train vs Validate)')
    plt.legend()
    if viz_filepath:
        plt.savefig(viz_filepath)
        print('viz saved to {}'.format(viz_filepath))
    if show_loss:
        plt.show()
    plt.clf()

def train_model(device, c_net, c_optimizer, c_loss_func, train_ds, validate_ds, hyperparameters):
    # Setup the train and validate data loaders.
    train = DataLoader(train_ds, hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn_pad)
    validate = DataLoader(validate_ds, int(hyperparameters['batch_size']/8), shuffle=True, collate_fn=collate_fn_pad)

    # Create the model, loss, and optimizer.
    net = c_net(model.NUM_FEATURES, model.NUM_PLANS).to(device)
    loss_func = c_loss_func()
    optimizer = c_optimizer(net.parameters(), lr=hyperparameters['lr'])

    # Loop through all epochs.
    train_loss = []
    validate_loss = []
    lowest_loss = None
    m_name = d_serialize(hyperparameters)
    for epoch in tqdm(range(hyperparameters['n_epochs'])):

        # Train the model.
        net.train()
        loss_sum = 0.0
        for batch_idx, data in enumerate(train):
            x, a, y = data
            x = x.to(device)
            a = a.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_est = net(x)[0]
            y_est = y_est[np.arange(0, y_est.shape[0]), a]
            loss = loss_func(y_est, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item()
        train_loss.append(loss_sum / (batch_idx+1))

        # Evaluate it.
        with torch.no_grad():
            net.eval()
            loss_sum = 0.0
            for batch_idx, data in enumerate(train):
                x, a, y = data
                x = x.to(device)
                a = a.to(device)
                y = y.to(device)
                y_est = net(x)[0]
                y_est = y_est[np.arange(0, y_est.shape[0]), a]
                loss = loss_func(y_est, y)
                loss_sum += loss.detach().item()
            validate_loss.append(loss_sum / (batch_idx+1))

        # Save off the model, if it is currently best and we're in the home stretch.
        if (hyperparameters['n_epochs'] - epoch) < 100:
            if lowest_loss is None or lowest_loss > train_loss[-1]:
                torch.save(net.state_dict(), os.path.join('models', m_name+'.pt'))
                lowest_loss = train_loss[-1]

    # Log model loss.
    train_loss = moving_average(train_loss, window_size=100)
    validate_loss = moving_average(validate_loss, window_size=100)
    viz_loss(train_loss, validate_loss, viz_filepath=os.path.join('models', m_name+'loss.png'))

def init_training():
    train_ds = SqlDataset(os.path.join('data', 'tr_data.json'))
    validate_ds = SqlDataset(os.path.join('data', 'val_data.json'))
    test_ds = SqlDataset(os.path.join('data', 'test_data.json'))
    hyperparameters = list(product_dict(**get_hyperparameters()))
    net = model.Lstm2XNetwork
    loss_func = torch.nn.MSELoss
    optimizer = torch.optim.AdamW
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device.type}")
    
    for i, h in enumerate(hyperparameters):
        print(f"{i+1}/{len(hyperparameters)}: Model n_epochs={h['n_epochs']} lr={h['lr']} batch_size={h['batch_size']}")
        train_model(device, net, optimizer, loss_func, train_ds, validate_ds, h)
    
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def d_serialize(d):
    s = ''
    for k, v in d.items():
        s += str(v)+'_'
    return s

def get_hyperparameters():
    return {
        'n_epochs': [500],
        'lr': [0.0001],
        'batch_size': [32]
    }

if __name__ == '__main__':
    init_training()