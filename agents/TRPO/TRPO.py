import abc
import copy
from abc import ABC

import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.autograd import Variable

from agents.NPG.NPG import NPG
from agents.config import TRPO_CONFIG


class TRPO(NPG, ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        config = copy.deepcopy(TRPO_CONFIG)
        config.update(parameters)
        super(TRPO, self).__init__(config)
        self.accept_ratio = config['accept_ratio']
        self.max_search_num = config['max_search_num']
        self.step_frac = config['step_frac']
        self.improvement = 0
        self.expected_improvement = 0
        self.env = config['env']
        self.max_steps = self.env.max_episode_steps
        self.cur_kl = 0

    def object_loss(self, theta):
        curr_policy = copy.deepcopy(self.policy)
        vector_to_parameters(theta, curr_policy.parameters())
        imp_fac = self.compute_ratio(curr_policy=curr_policy)
        loss = - (imp_fac * self.A).mean()
        current_kl = self.mean_kl_divergence(model=curr_policy)
        return loss, current_kl

    # completely copied, yet to be understood
    def linear_search(self, theta, fullstep, expected_improve_rate):
        accept_ratio = self.accept_ratio
        max_backtracks = self.max_search_num
        loss_func_val, current_kl = self.object_loss(theta)
        print("*****************************************")
        for (_n_backtracks, stepfrac) in enumerate(list(self.step_frac ** torch.arange(0, max_backtracks).float().type_as(self.state))):
            theta_new = theta + stepfrac * fullstep
            new_loss_func_val, current_kl = self.object_loss(theta_new)
            actual_improve = loss_func_val - new_loss_func_val
            expected_improve = expected_improve_rate * stepfrac
            if not expected_improve.item() >= 1 and not expected_improve.item() < 1:
                pass
            ratio = actual_improve / expected_improve
            print("Search number {}...".format(_n_backtracks + 1) + " Actual improve: {:.5f}".format(actual_improve) +
                  " Expected improve: {:.5f}".format(expected_improve) + " Current KL: {:.8f}".format(current_kl))
            if ratio.item() > accept_ratio and actual_improve.item() > 0 and 0 < current_kl < self.max_kl * 1.5:
                self.improvement = actual_improve.item()
                self.expected_improvement = expected_improve.item()
                print("*****************************************")
                return theta_new
        print("** Failure optimization, rolling back. **")
        print("*****************************************")
        return theta

    def update_policy(self):
        self.n_traj = 0
        self.sample_batch()
        self.split_episode()

        self.other_data = self.goal
        self.estimate_value()
        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-8)  # TODO: Why uniforming advantages?
        likelihood_ratio = self.compute_ratio()

        self.L = - (likelihood_ratio * self.A).mean()   # TODO: I changed self.A.squeeze() to self.A
        if self.value_func_type is not None:
            # update value for train_v_iters times
            for i in range(self.train_v_iters):
                self.update_value()
        self.policy.zero_grad()
        L_grad = torch.autograd.grad(self.L, self.policy.parameters(), create_graph=True)
        # turn the gradient of loss into a 1-D Variable
        L_grad_vector = parameters_to_vector([g for g in L_grad])
        # solve Hx = -g i.e. H^(-1)g = x, H is the Hessian Matrix of KL divergence
        Hg = self.conjugate_gradient(-L_grad_vector)
        gHg = .5 * torch.sum(- L_grad_vector * Hg).detach() # 1/2 * gHg
        sqrt_coef = torch.sqrt(self.max_kl / gHg)
        gdotstepdir = -torch.sum(L_grad_vector * Hg)    # -gHg
        # This is where different than NPG
        # param_new = parameters_to_vector(self.policy.parameters()) + sqrt_coef * Hg
        param_new = self.linear_search(parameters_to_vector(
            self.policy.parameters()), sqrt_coef * Hg, gdotstepdir * sqrt_coef) # -gHg*sqrt(2 * delta / gHg)
        vector_to_parameters(param_new, self.policy.parameters())
        self.learn_step_counter += 1
        self.cur_kl = self.mean_kl_divergence().item()

