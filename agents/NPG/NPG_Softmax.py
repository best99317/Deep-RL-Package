import numpy as np
import torch
from .NPG import NPG
from agents.VPG import VPG_Softmax


class NPG_Softmax(NPG, VPG_Softmax):
    def __init__(self, parameters):
        super(NPG_Softmax, self).__init__(parameters)

    def mean_kl_divergence(self, inds=None, model=None):
        if inds is None:
            inds = np.arange(self.state.size(0))
        if model is None:
            model = self.policy
        distri_1 = model(self.state[inds], other_data=self.other_data[inds])
        distri_2 = self.distribution[inds].squeeze()
        log_ratio = torch.log(distri_2 / distri_1)
        mean_kl = torch.sum(distri_2 * log_ratio, 1).mean()
        return mean_kl
