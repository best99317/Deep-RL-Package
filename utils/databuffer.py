import abc
import copy
import numpy as np
from .config import DATABUFFER_CONFIG

class databuffer(object):
    def __init__(self, hyperparams):
        config = copy.deepcopy(DATABUFFER_CONFIG)
        config.update(hyperparams)
        self.max_size = config['memory_size']
        self.state_dims = config['num_states']
        if isinstance(self.state_dims, dict):
            self.state_dims = self.state_dims['state_dim']
        if 'num_actions' in config.keys():
            self.num_actions = config['num_actions']
        self.actions_dims = config['action_dims']
        self.discrete_action = config['discrete_action']
        if isinstance(self.state_dims, int):
            self.state_dims = (self.state_dims, )
        self.S = np.zeros((0,) + self.state_dims, dtype = np.float32)
        self.A = np.zeros([0, self.actions_dims], dtype = np.uint8 if self.discrete_action else np.float32)
        self.R = np.zeros([0, 1], dtype = np.float32)
        self.S_ = np.zeros((0,) + self.state_dims, dtype = np.float32)
        self.done = np.zeros([0, 1], dtype = np.uint8)

        # other data in transitions. For example, goals, episode infos, etc.
        self.other_data = None
        if 'other_data' in config:
            self.other_data = {}
            for key, box in config['other_data'].items():
                self.other_data[key] = np.zeros((0,) + box.shape[1:], dtype=box.dtype)

        # memory counter: How many transitions are recorded in total
        self.mem_c = 0

    def store_transition(self, transitions):
        self.S = np.concatenate((self.S, transitions['state']), axis=0)
        self.A = np.concatenate((self.A, transitions['action']), axis=0)
        self.R = np.concatenate((self.R, transitions['reward']), axis=0)
        self.done = np.concatenate((self.done, transitions['done']), axis=0)
        self.S_ = np.concatenate((self.S_, transitions['next_state']), axis=0)
        if self.other_data:
            for key in self.other_data.keys():
                assert 'other_data' in transitions, \
                    "Other data types should be included in transitions except S, A, R, Done, and S_."
                self.other_data[key] = np.concatenate((self.other_data[key], transitions['other_data'][key]), axis=0)
        self.mem_c += transitions['state'].shape[0]
        if self.mem_c >self.max_size:
            self.S = self.S[-self.max_size:]
            self.A = self.A[-self.max_size:]
            self.R = self.R[-self.max_size:]
            self.done = self.done[-self.max_size:]
            self.S_ = self.S_[-self.max_size:]
            if self.other_data:
                for key in self.other_data.keys():
                    self.other_data[key] = self.other_data[key][-self.max_size:]

    def sample_batch(self, batch_size = None):
        if batch_size is not None:
            if batch_size > self.mem_c or batch_size > self.max_size:
                raise RuntimeError("Batch size is bigger than buffer size")
            # sample without putting back
            # sample_index = np.random.choice(min(self.max_size, self.mem_c), size=batch_size)
            # sample with putting back
            sample_index = np.random.randint(0, self.mem_c, size=batch_size)
        else:
            sample_index = np.arange(min(self.max_size, self.mem_c))
        batch = {}
        batch['state'] = self.S[sample_index]
        batch['action'] = self.A[sample_index]
        batch['reward'] = self.R[sample_index]
        batch['done'] = self.done[sample_index]
        batch['next_state'] = self.S_[sample_index]
        batch['other_data'] = None
        if self.other_data:
            batch['other_data'] = {}
            for key in self.other_data.keys():
                batch['other_data'][key] = self.other_data[key][sample_index]
        return batch, sample_index

    def reset_buffer(self):
        self.S = np.zeros((0,) + self.state_dims, dtype=np.float32)
        self.A = np.zeros([0, self.actions_dims], dtype = np.uint8 if self.discrete_action else np.float32)
        self.R = np.zeros([0, 1], dtype = np.float32)
        self.S_ = np.zeros((0,) + self.state_dims, dtype=np.float32)
        self.done = np.zeros([0, 1], dtype = np.bool)
        if self.other_data:
            for key in self.other_data.keys():
                self.other_data[key] = np.zeros((0,) + self.other_data[key].shape[1:], dtype=self.other_data[key].dtype)
        self.mem_c = 0

class databuffer_PG_gaussian(databuffer):
    def __init__(self, hyperparams):
        super(databuffer_PG_gaussian, self).__init__(hyperparams)
        self.mu = np.zeros([0, self.actions_dims])
        self.sigma = np.zeros([0, self.actions_dims])
        self.logpi = np.zeros([0, 1], dtype=np.float32)

    def store_transition(self, transitions):
        databuffer.store_transition(self, transitions)
        self.mu = np.concatenate((self.mu, transitions['mu']), axis=0)
        self.sigma = np.concatenate((self.sigma, transitions['sigma']), axis=0)
        self.logpi = np.concatenate((self.logpi, transitions['logpi']), axis=0)
        if self.mem_c >self.max_size:
            self.mu = self.mu[-self.max_size:]
            self.sigma = self.sigma[-self.max_size:]
            self.logpi = self.logpi[-self.max_size:]

    def sample_batch(self, batch_size= None):
        batch, sample_index = databuffer.sample_batch(self, batch_size)
        batch['mu'] = self.mu[sample_index]
        batch['sigma'] = self.sigma[sample_index]
        batch['logpi'] = self.logpi[sample_index]
        return batch, sample_index

    def reset_buffer(self):
        databuffer.reset_buffer(self)
        self.mu = np.zeros([0, self.actions_dims])
        self.sigma = np.zeros([0, self.actions_dims])
        self.logpi = np.zeros([0, 1], dtype=np.float32)

class databuffer_PG_softmax(databuffer):
    def __init__(self, hyperparams):
        super(databuffer_PG_softmax, self).__init__(hyperparams)
        self.distribution = np.zeros([0, self.num_actions])
        self.logpi = np.zeros([0, 1], dtype=np.float32)

    def store_transition(self, transitions):
        databuffer.store_transition(self, transitions)
        self.distribution = np.concatenate((self.distribution, transitions['distribution']), axis=0)
        self.logpi = np.concatenate((self.logpi, transitions['logpi']), axis=0)
        if self.mem_c >self.max_size:
            self.distribution = self.distribution[-self.max_size:]
            self.logpi = self.logpi[-self.max_size:]

    def sample_batch(self, batch_size= None):
        batch, sample_index = databuffer.sample_batch(self, batch_size)
        batch['distribution'] = self.distribution[sample_index]
        batch['logpi'] = self.logpi[sample_index]
        return batch, sample_index

    def reset_buffer(self):
        databuffer.reset_buffer(self)
        self.distribution = np.zeros([0, self.num_actions])
        self.logpi = np.zeros([0, 1], dtype=np.float32)