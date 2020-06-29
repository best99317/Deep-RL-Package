from abc import ABC

import torch
import numpy as np
from agents.HTRPO.HTRPO import HTRPO
from agents.HPG.HPG_Gaussian import HPG_Gaussian


class HTRPO_Gaussian(HTRPO, HPG_Gaussian, ABC):
    def __init__(self, parameters):
        super(HTRPO_Gaussian, self).__init__(parameters)

    def mean_kl_divergence(self, inds=None, model=None):
        if inds is None:
            inds = np.arange(self.state.size(0))
        if model is None:
            model = self.policy
        mu1, log_sigma1, sigma1 = model(self.state[inds], other_data=self.other_data[inds])
        logp = self.compute_logp(mu1, log_sigma1, sigma1, self.action[inds])
        logp_old = self.logpi_old[inds].squeeze()
        mean_kl = (1 - self.gamma) * torch.sum(
            self.hratio[inds] * self.gamma_discount * 0.5 * torch.pow((logp_old - logp), 2)) / self.n_traj
        return mean_kl