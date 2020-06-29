import abc
import copy
from abc import ABC
import numpy as np
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from utils.vecenv import space_dim
from utils.rms import RunningMeanStd
from agents.TRPO.TRPO import TRPO
from agents.config import HTRPO_CONFIG
from agents.HPG.HPG import HPG


class HTRPO(HPG, TRPO, ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        config = copy.deepcopy(HTRPO_CONFIG)
        config.update(parameters)
        super(HTRPO, self).__init__(config)
        self.env = config['env']
        self.max_steps = self.env.max_episode_steps

        self.sampled_goal_num = config['sampled_goal_num']
        self.subgoals = None
        self.goal_space = config['goal_space']
        self.dim_goal = space_dim(self.goal_space)
        self.weight_is = config['weighted_is']
        self.per_decision = config['per_decision'] # TODO: What is per_decision?
        self.entropy_weight = config['entropy_weight']
        if self.norm_ob:
            self.ob_rms = {}
            for key in self.env.observation_space.spaces.keys():
                self.ob_rms[key] = RunningMeanStd(shape=self.env.observation_space.spaces[key].shape)
            self.ob_mean = [0,]
            self.ob_var = [1,]
            self.goal_mean = [0,]
            self.goal_var = [1,]

    def compute_entropy(self, inds=None, model=None):
        # default: compute all importance factors.
        if inds is None:
            inds = np.arange(self.state.size(0))
        if model is None:
            model = self.policy
        if self.discrete_action:
            distribution = model(self.state[inds],
                                            other_data=self.other_data[inds] if self.other_data is not None else None)
            entropy = - torch.sum(distribution * torch.log(distribution), 1).mean()
        else:
            mu_now, logsigma_now, _ = model(self.state[inds],
                                            other_data=self.other_data[inds] if self.other_data is not None else None)
            entropy = (0.5 * self.action_dims * np.log(2 * np.pi * np.e) + torch.sum(logsigma_now, 1)).mean()
        return entropy

    def update_policy(self):
        self.n_traj = 0
        self.sample_batch()
        self.split_episode()

        self.generate_subgoals()
        self.reset_training_data()
        if self.sampled_goal_num is None or self.sampled_goal_num > 0 :
            self.generate_fake_data()

        if self.norm_ob:
            self.ob_rms['observation'].update(self.state.cpu().numpy())
            self.ob_rms['desired_goal'].update(self.goal.cpu().numpy())

        if self.norm_ob:
            self.state = torch.clamp((self.state - torch.Tensor(self.ob_mean).type_as(self.state).unsqueeze(0)) /
                         torch.sqrt(torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(self.state).unsqueeze(0)),
                                     -5, 5)
            self.goal = torch.clamp((self.goal - torch.Tensor(self.goal_mean).type_as(self.state).unsqueeze(0)) /
                        torch.sqrt(torch.clamp(torch.Tensor(self.goal_var), 1e-4).type_as(self.state).unsqueeze(0)),
                                    -5, 5)

        self.other_data = self.goal

        self.estimate_value()
        if self.value_func_type is not None:
            # update value
            for i in range(self.train_v_iters):
                self.update_value()

        likelihood_ratio = self.compute_ratio()
        if self.value_func_type:
            # old value estimator
            self.A = self.gamma_discount * self.hratio * self.A
        else:
            self.A = self.gamma_discount * self.A

        # self.L = - likelihood_ratio * self.A
        #
        # L_grad = [None] * len(self.L)
        #
        # indexes = self.hratio.gt(self.c) | self.hratio.lt(1. / self.c)
        #
        # for i in range(len(self.L)):
        #     if not indexes[i]:
        #         L_grad[i] = torch.autograd.grad(self.L[i], self.policy.parameters(), create_graph=True)
        #     else:
        #         L_grad[i] = torch.zeros().type_as(self.policy.parameters())
        #
        # L_grad = torch.stack(L_grad).mean(0)

        self.L = - (likelihood_ratio * self.A).mean()
        # Add Entropy Bonus
        self.L = self.L - self.entropy_weight * self.compute_entropy()
        self.policy.zero_grad()
        L_grad = torch.autograd.grad(self.L, self.policy.parameters(), create_graph=True)
        # turn the gradient of loss into a 1-D Variable

        L_grad_vector = parameters_to_vector([g for g in L_grad])

        # solve Hx = -g i.e. H^(-1)g = x, H is the Hessian Matrix of KL divergence
        Hg = self.conjugate_gradient(-L_grad_vector)
        gHg = .5 * torch.sum(- L_grad_vector * Hg).detach()  # 1/2 * gHg
        sqrt_coef = torch.sqrt(self.max_kl / gHg)
        gdotstepdir = -torch.sum(L_grad_vector * Hg)  # -gHg
        # This is where different than NPG
        # param_new = parameters_to_vector(self.policy.parameters()) + sqrt_coef * Hg
        param_new = self.linear_search(parameters_to_vector(
            self.policy.parameters()), sqrt_coef * Hg, gdotstepdir * sqrt_coef)  # -gHg*sqrt(2 * delta / gHg)
        vector_to_parameters(param_new, self.policy.parameters())
        self.learn_step_counter += 1
        self.cur_kl = self.mean_kl_divergence().item()

        if self.norm_ob:
            self.ob_mean = self.ob_rms['observation'].mean
            self.ob_var = self.ob_rms['observation'].var
            self.goal_mean = self.ob_rms['desired_goal'].mean
            self.goal_var = self.ob_rms['desired_goal'].var
