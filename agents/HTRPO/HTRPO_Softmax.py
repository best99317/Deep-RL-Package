import torch
import numpy as np
from agents.HTRPO.HTRPO import HTRPO
from agents.HPG.HPG_Softmax import HPG_Softmax


class HTRPO_Softmax(HTRPO, HPG_Softmax):
    def __init__(self, parameters):
        super(HTRPO_Softmax, self).__init__(parameters)

    def mean_kl_divergence(self, inds=None, model=None):
        if inds is None:
            inds = np.arange(self.state.size(0))
        if model is None:
            model = self.policy
        distri1 = model(self.state[inds], other_data=self.other_data[inds])
        logp = self.compute_logp(distri1, self.action[inds])
        logp_old = self.logpi_old[inds].squeeze()
        mean_kl = (1 - self.gamma) * torch.sum(
            self.hratio[inds] * self.gamma_discount * 0.5 * torch.pow((logp_old - logp), 2)) / self.n_traj
        return mean_kl
