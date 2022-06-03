import json
import os
import configparser

class ObservationStore():

    def __init__(self, f_name=os.path.join('data', 'training.json')):
        config = configparser.ConfigParser()
        config.read('server.cfg')
        cfg = config['buffer']
        self.flush_after = int(cfg['FlushAfter'])
        self.f_name = f_name
        self.buffer = [None for _ in range(self.flush_after)]
        self.i_buffer = 0
        self.stage_buffer = {}
        self.stage_last = None

    def stage(self, state, action, key=None):
        self.stage_last = (state, action)
        #self.stage_buffer[key] = (state, action)

    def record(self, reward, key=None):
        if self.stage_last:
            state, action = self.stage_last#self.stage_buffer.pop(key)
            self.buffer[self.i_buffer] = {'x': state, 'a': action, 'y': reward}
            self.i_buffer += 1
            if self.i_buffer >= self.flush_after:
                self.flush_buffer()

    def flush_buffer(self):
        with open(self.f_name, 'a') as f_train:
            for obs in self.buffer:
                json.dump(obs, f_train)
                f_train.write(',')
        self.i_buffer = 0