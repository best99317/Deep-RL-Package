import numpy as np
import torch
from .NPG import NPG
from agents.VPG import VPG_Gaussian


class NPG_Gaussian(NPG, VPG_Gaussian):
    def __init__(self, parameters):
        super(NPG_Gaussian, self).__init__(parameters)

    def mean_kl_divergence(self, inds=None, model=None):
        if inds is None:
            inds = np.arange(self.state.size(0))
        if model is None:
            model = self.policy
        mu1, log_sigma1, sigma1 = model(self.state[inds],
                                        other_data=self.other_data[inds] if self.other_data is not None else None)
        if self.using_KL_estimation:
            logpi = self.compute_logp(mu1, log_sigma1, sigma1, self.action[inds])
            logpi_old = self.logpi_old[inds].squeeze()
            mean_kl = 0.5 * torch.mean(torch.pow((logpi_old - logpi),2))
        else:
            mu2, log_sigma2, sigma2 = self.mu[inds], torch.log(self.sigma[inds]), self.sigma[inds]
            sigma1 = torch.pow(sigma1, 2)
            sigma2 = torch.pow(sigma2, 2)
            mean_kl = 0.5 * (torch.sum(torch.log(sigma1) - torch.log(sigma2), dim=1) - self.action_dims +
                        torch.sum(sigma2 / sigma1, dim=1) + torch.sum(torch.pow((mu1 - mu2), 2) / sigma1, 1)).mean()
        return mean_kl
