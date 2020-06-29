from abc import ABC

import torch
import copy
import numpy as np

from .VPG import VPG, VPG_CONFIG
from utils.databuffer import databuffer_PG_softmax
from basenets.base_PG import FCPG_Softmax, ConvPG_Softmax

#########################
# Define Policy Network #
#########################


class VPG_Softmax(VPG, ABC):
    def __init__(self, parameters):
        config = copy.deepcopy(VPG_CONFIG)
        config.update(parameters)
        super(VPG_Softmax, self).__init__(config)

        # initialize memory buffer
        self.memory_size = config['memory_size']
        self.memory = databuffer_PG_softmax(config)
        self.sample_index = None

        # initialize policy
        if self.policy_type == 'FC':
            self.policy = FCPG_Softmax(
                self.num_states['state_dim'] + self.num_states['goal_dim'] \
                            if isinstance(self.num_states, dict) else self.num_states,
                self.num_actions,
                num_hiddens=self.hidden_layers,
                batch_normalization=self.batch_normalization,
                activ_func=self.activ_func,
            )
        elif self.value_func_type == 'Conv':
            self.policy = ConvPG_Softmax(
                self.num_states,
                self.num_actions,
                fcs=self.hidden_layers,
                activ_func=self.activ_func,
                batch_normalization=self.batch_normalization,
            )

        # initialize optimizer_pi
        self.momentum_pi = config['momentum_pi']
        if self.momentum_pi is not None:
            self.optimizer_pi = config['optimizer_pi'](self.policy.parameters(), lr=self.lr_pi,
                                                     momentum=self.momentum_pi)
        else:
            self.optimizer_pi = config['optimizer_pi'](self.policy.parameters(), lr=self.lr_pi)

        # initialize distribution
        self.distribution = torch.Tensor(1)

    # use cuda
    def cuda(self):
        VPG.cuda(self)
        self.distribution = self.distribution.cuda()

    def choose_action(self, state, other_data=None, greedy=False):
        self.policy.eval()
        if self.use_cuda:
            state = state.cuda()
            if other_data is not None:
                other_data = other_data.cuda()
        distribution = self.policy(state, other_data).detach()
        self.policy.train()
        # TODO: Why not choose action greedy? How to train if not so?
        if not greedy:
            action = torch.multinomial(distribution, 1, replacement=True)
        else:
            _, action = torch.max(distribution, dim=-1, keepdim=True)
        return action, distribution

    def compute_logp(self, distribution, action):
        if distribution.dim() == 1:
            return torch.log(distribution[action] + 1e-10)
        elif distribution.dim() == 2:
            action_indexes = [range(distribution.size(0)), action.squeeze(-1).long().cpu().numpy().tolist()]
            return torch.log(distribution[action_indexes] + 1e-10)
        else:
            RuntimeError("distribution must be a 1-D or 2-D Tensor or Variable")

    def compute_ratio(self, indexes=None, curr_policy=None):
        if indexes is None:
            indexes = np.arange(self.state.size(0))
        if curr_policy is None:
            curr_policy = self.policy
        distribution = curr_policy(self.state[indexes], other_data=self.other_data[indexes]
                                                        if self.other_data is not None else None)
        likelihood_ratio = torch.exp(self.compute_logp(distribution, self.action[indexes])
                                     - self.logpi_old[indexes].squeeze())
        return likelihood_ratio

    def mean_kl_divergence(self, inds=None, model=None):
        if inds is None:
            inds = np.arange(self.state.size(0))
        if model is None:
            model = self.policy
        distri1 = model(self.state[inds], other_data=self.other_data[inds])
        distri2 = self.distribution[inds].squeeze()
        log_ratio = torch.log(distri2 / distri1)
        mean_kl = torch.sum(distri2 * log_ratio, 1).mean()
        return mean_kl