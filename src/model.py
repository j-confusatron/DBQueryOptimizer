import torch
import os
import random
import configparser

NUM_PLANS = 5
NUM_FEATURES = 15 * NUM_PLANS

class LstmNetwork(torch.nn.Module):

    def __init__(self, n_input, n_output):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.lstm = torch.nn.LSTM(n_input, 1024, bidirectional=True)
        self.linears = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, n_output)
        )
    
    def forward(self, x):
        _, h_c = self.lstm(x)
        h = torch.sum(h_c[0], 0)
        y = self.linears(h)
        return y, h_c


class LstmXLNetwork(torch.nn.Module):

    def __init__(self, n_input, n_output):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.lstm = torch.nn.LSTM(n_input, 1024, num_layers=2, bidirectional=True)
        self.linears = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, n_output)
        )
    
    def forward(self, x):
        ltsm_out, h_c = self.lstm(x)
        fc_in = ltsm_out[-1,:,:]
        y = self.linears(fc_in)
        return y, h_c


class Lstm2XNetwork(torch.nn.Module):

    def __init__(self, n_input, n_output):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.lstm = torch.nn.LSTM(n_input, 2048, num_layers=2, bidirectional=True)
        self.linears = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, n_output)
        )
    
    def forward(self, x):
        ltsm_out, h_c = self.lstm(x)
        fc_in = ltsm_out[-1,:,:]
        y = self.linears(fc_in)
        return y, h_c


class Lstm3XNetwork(torch.nn.Module):

    def __init__(self, n_input, n_output):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.lstm = torch.nn.LSTM(n_input, 2048, num_layers=3, bidirectional=True)
        self.linears = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, n_output)
        )
    
    def forward(self, x):
        ltsm_out, h_c = self.lstm(x)
        fc_in = ltsm_out[-1,:,:]
        y = self.linears(fc_in)
        return y, h_c


class Model():

    def __init__(self, model_name):
        config = configparser.ConfigParser()
        config.read('server.cfg')
        cfg = config['model']

        self.model = Lstm2XNetwork(NUM_FEATURES, NUM_PLANS)

        m_path = os.path.join('models', model_name)
        if os.path.exists(m_path):
            self.model.load_state_dict(torch.load(m_path))

        self.model.eval()
    
        self.epsilon = float(cfg['Epsilon'])
        self.force_arm = int(cfg['ForceArm'])
        
    def select_plan(self, plan):
        if self.force_arm > -1:
            y = self.force_arm

        elif random.random() > self.epsilon:
            x = torch.unsqueeze(torch.tensor(plan), 0)
            y, h_c = self.model(x)
            y = torch.argmax(y, 1).item()

        else:
            y = random.randrange(NUM_PLANS)

        return y

    def predict(self, plan):
        # Not currently used.
        x = torch.unsqueeze(torch.tensor(plan), 0)
        y, h_c = self.model(x)
        return y