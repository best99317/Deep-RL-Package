from abc import ABC

import torch
import copy
import numpy as np

from .VPG import VPG, VPG_CONFIG
from utils.databuffer import databuffer_PG_gaussian
from basenets.base_PG import FCPG_Gaussian


#########################
# Define Policy Network #
#########################


class VPG_Gaussian(VPG, ABC):
    def __init__(self, parameters):
        config = copy.deepcopy(VPG_CONFIG)
        config.update(parameters)
        super(VPG_Gaussian, self).__init__(config)

        # initialize memory buffer
        self.memory_size = config['memory_size']
        self.memory = databuffer_PG_gaussian(config)
        self.sample_index = None

        # initialize policy
        self.action_bounds = config['action_bounds']
        self.init_noise = config['init_noise']

        if self.policy_type == 'FC':
            self.policy = FCPG_Gaussian(
                self.num_states['state_dim'] + self.num_states['goal_dim'] \
                    if isinstance(self.num_states, dict) else self.num_states,
                self.action_dims,
                out_active=self.out_activ_func,
                num_hiddens=self.hidden_layers,
                batch_normalization=self.batch_normalization,
                activ_func=self.activ_func,
                sigma=self.init_noise,
                out_scaler=self.action_bounds
            )
        elif self.value_func_type == 'Conv':
            raise NotImplementedError

        # initialize optimizer_pi
        self.momentum_pi = config['momentum_pi']
        if self.momentum_pi is not None:
            self.optimizer_pi = config['optimizer_pi'](self.policy.parameters(), lr=self.lr_pi,
                                                       momentum=self.momentum_pi)
        else:
            self.optimizer_pi = config['optimizer_pi'](self.policy.parameters(), lr=self.lr_pi)

        # initialize distribution
        self.mu = torch.Tensor(1)
        self.sigma = torch.Tensor(1)

    # use cuda
    def cuda(self):
        VPG.cuda(self)
        self.mu = self.mu.cuda()
        self.sigma = self.sigma.cuda()

    def choose_action(self, state, other_data=None, greedy=False):
        self.policy.eval()
        if self.use_cuda:
            state = state.cuda()
            if other_data is not None:
                other_data = other_data.cuda()
        mu, log_sigma, sigma = self.policy(state, other_data)
        mu = mu.detach()
        log_sigma = log_sigma.detach()
        sigma = sigma.detach()
        self.policy.train()
        # TODO: Why not choose action greedy? How to train if not so?
        if not greedy:
            action = torch.normal(mu, sigma)
        else:
            action = mu
        return action, mu, log_sigma, sigma

    # TODO: Why break up the log of Gaussian into this complex form
    # TODO: WHy consider 1 and 2 action.dim()?
    # TODO: Why is self.n_action_dims used for computing?
    def compute_logp(self, mu, log_sigma, sigma, action):
        if action.dim() == 1:
            return -0.5 * torch.sum(torch.pow((action - mu) / sigma, 2)) \
                   - 0.5 * self.action_dims * torch.log(torch.Tensor([2. * np.pi]).type_as(mu)) \
                   - torch.sum(log_sigma)
        elif action.dim() == 2:
            return -0.5 * torch.sum(torch.pow((action - mu) / sigma, 2), 1) \
                   - 0.5 * self.action_dims * torch.log(torch.Tensor([2. * np.pi]).type_as(mu)) \
                   - torch.sum(log_sigma, 1)
        else:
            RuntimeError("distribution must be a 1-D or 2-D Tensor or Variable")

    def compute_ratio(self, indexes=None, curr_policy=None):
        if indexes is None:
            indexes = np.arange(self.state.size(0))
        if curr_policy is None:
            curr_policy = self.policy
        curr_mu, curr_log_sigma, curr_sigma = curr_policy(self.state[indexes], other_data=self.other_data[indexes]
        if self.other_data is not None else None)
        likelihood_ratio = torch.exp(self.compute_logp(curr_mu, curr_log_sigma, curr_sigma, self.action[indexes])
                                     - self.logpi_old[indexes].squeeze())
        return likelihood_ratio

    def mean_kl_divergence(self, inds=None, model=None):
        if inds is None:
            inds = np.arange(self.state.size(0))
        if model is None:
            model = self.policy
        mu1, log_sigma1, sigma1 = model(self.state[inds], other_data=self.other_data[inds])
        mu2, logsigma2, sigma2 = self.mu[inds], torch.log(self.sigma[inds]), self.sigma[inds]
        sigma1 = torch.pow(sigma1, 2)
        sigma2 = torch.pow(sigma2, 2)
        mean_kl = 0.5 * (torch.sum(torch.log(sigma1) - torch.log(sigma2), dim=1) - self.action_dims +
                    torch.sum(sigma2 / sigma1, dim=1) + torch.sum(torch.pow((mu1 - mu2), 2) / sigma1, 1)).mean()
        return mean_kl
