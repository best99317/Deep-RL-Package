import abc
import copy

import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.autograd import Variable

from agents.VPG.VPG import VPG
from agents.config import NPG_CONFIG


class NPG(VPG):
    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        config = copy.deepcopy(NPG_CONFIG)
        config.update(parameters)
        super(NPG, self).__init__(config)
        self.cg_iters = config['cg_iters']
        self.cg_residual_tol = config['cg_residual_tol']  # TODO: What does cg_residual_tol mean?
        self.cg_damping = config['cg_damping']
        self.max_kl = config['max_kl_divergence']

    def conjugate_gradient(self, b):
        """
                Demmel p 312, borrowed from https://github.com/ikostrikov/pytorch-trpo
        """
        p = b.clone().data
        r = b.clone().data
        x = torch.zeros_like(b).data
        r_two_norm = torch.sum(r * r)
        for i in range(self.cg_iters):
            Ap = self.hessian_vector_product(Variable(p))
            alpha = r_two_norm / torch.sum(p * Ap.data)
            x += alpha * p
            r -= alpha * Ap.data
            new_r_two_norm = torch.sum(r * r)
            beta = new_r_two_norm / r_two_norm
            p = r + beta * p
            r_two_norm = new_r_two_norm
            if r_two_norm < self.cg_residual_tol:
                break
        return Variable(x)

    # Implemented in Softmax and Gaussian respectively
    @abc.abstractmethod
    def mean_kl_divergence(self, inds=None, model=None):
        raise NotImplementedError("Must be implemented in subclass.")

    # This function return the product of Hessian and the input vector
    def hessian_vector_product(self, vector):
        """
        Returns the product of the Hessian of the KL divergence and the given vector
        Borrowed from https://github.com/mjacar/pytorch-trpo/blob/master/trpo_agent.py
        """
        self.policy.zero_grad()
        # compute KL divergence
        mean_kl_div = self.mean_kl_divergence()
        # first order derivative of KL
        kl_grad = torch.autograd.grad(mean_kl_div, self.policy.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])  # TODO: don't know what torch.cat() does
        # d(KL) * vector
        grad_vector_product = torch.sum(kl_grad_vector * vector)
        # second order d(d(KL)*vector)
        grad_grad = torch.autograd.grad(grad_vector_product, self.policy.parameters())
        fisher_vector_product = torch.cat([grad.contiguous().view(-1) for grad in grad_grad])
        return fisher_vector_product + (self.cg_damping * vector)   # TODO: What does cg_damping do?

    def update_policy(self):
        self.n_traj = 0  # refer to learn_trpo
        self.sample_batch()
        self.split_episode()
        self.other_data = self.goal
        self.estimate_value()
        # update policy
        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-8)
        likelihood_ratio = self.compute_ratio()
        self.L = - (likelihood_ratio * self.A.squeeze()).mean()

        # update value
        if self.value_func_type is not None:
            # update value for train_v_iters times
            for i in range(self.train_v_iters):
                self.update_value()
        self.policy.zero_grad()
        # compute policy gradient g
        # TODO: torch.autograd.grad(L) vs. L.backword()?
        L_grad = torch.autograd.grad(self.L, self.policy.parameters(), create_graph=True)
        # turn the gradient of loss into a 1-D Variable
        L_grad_vector = parameters_to_vector([g for g in L_grad])
        # solve Hx = -g i.e. H^(-1)g = x, H is the Hessian Matrix of KL divergence
        Hg = self.conjugate_gradient(-L_grad_vector)
        gHg = .5 * torch.sum(- L_grad_vector * Hg).detach()
        sqrt_coef = torch.sqrt(self.max_kl / gHg)
        param_new = parameters_to_vector(self.policy.parameters()) + sqrt_coef * Hg
        vector_to_parameters(param_new, self.policy.parameters())
        self.learn_step_counter += 1
        self.cur_kl = self.mean_kl_divergence().item()
