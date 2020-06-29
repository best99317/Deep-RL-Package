from abc import ABC

import torch
from torch import nn
import abc
import copy
from agents.VPG.VPG import VPG
from agents.config import HPG_CONFIG
from utils.vecenv import space_dim
from utils.rms import RunningMeanStd
import pickle
import os
import random
import numpy as np


class HPG(VPG, ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(HPG_CONFIG)
        config.update(hyperparams)
        super(HPG, self).__init__(config)
        self.env = config['env']
        self.max_steps = self.env.max_episode_steps
        self.norm_ob = config['norm_ob']  # do normalization for observation
        self.sampled_goal_num = config['sampled_goal_num']
        self.subgoals = None
        self.goal_space = config['goal_space']
        self.dim_goal = space_dim(self.goal_space)
        self.weight_is = config['weighted_is']
        self.per_decision = config['per_decision']  # TODO: What is per_decision?
        self.hratio = 0
        if self.norm_ob:
            self.ob_rms = {}
            for key in self.env.observation_space.spaces.keys():
                self.ob_rms[key] = RunningMeanStd(shape=self.env.observation_space.spaces[key].shape)
            self.ob_mean = [0, ]
            self.ob_var = [1, ]
            self.goal_mean = [0, ]
            self.goal_var = [1, ]
        self.norm_rw = config['norm_rw']
        self.ret = None
        if self.norm_rw:
            self.ret_rms = RunningMeanStd(shape=())
            self.rw_var = 1.

    def generate_fake_data(self):
        raise NotImplementedError("Must be implemented in subclasses")

    def generate_subgoals(self):
        achieved_goals = self.achieved_goal.cpu().numpy()
        self.subgoals = np.unique(achieved_goals.round(decimals=2), axis=0)

        if self.sampled_goal_num is not None:
            dg = np.unique(self.desired_goal.cpu().numpy().round(decimals=2), axis=0)
            dg_max = np.max(dg, axis=0)
            dg_min = np.min(dg, axis=0)

            g_ind = (dg_min != dg_max)
            # choose subgoals whose x and y fall in the range of min and max desired goals
            subgoals = self.subgoals[np.sum((self.subgoals[:, g_ind] > dg_max[g_ind]) |
                                            (self.subgoals[:, g_ind] < dg_min[g_ind]), axis=-1) == 0]

            # if all goals fall out of the range of desired_goals, choose ones with the min distance to desired_goals
            if subgoals.shape[0] == 0:
                dist_to_dg_center = np.linalg.norm(self.subgoals - np.mean(dg, axis=0), axis=1)
                ind_subgoals = np.argsort(dist_to_dg_center)
                self.subgoals = np.unique(np.concatenate([
                    self.subgoals[ind_subgoals[:self.sampled_goal_num]], subgoals
                ], axis=0), axis=0)
            else:
                self.subgoals = subgoals

            size_of_subgoals = min(self.sampled_goal_num, self.subgoals.shape[0])

            # initialize a random goal from achieved goals as initial goal
            init_index = np.random.randint(self.subgoals.shape[0])
            selected_subgoals = self.subgoals[init_index: init_index + 1]
            # exclude this selected goal from subgoals
            self.subgoals = np.delete(self.subgoals, init_index, axis=0)

            # compute the distance to the selected goal for the rest goals
            dists = np.linalg.norm(
                np.expand_dims(selected_subgoals, axis=0) - np.expand_dims(self.subgoals, axis=1),
                axis=-1)

            for g in range(size_of_subgoals - 1):
                # select the goal with max min distance to the other goals
                selected_ind = np.argmax(np.min(dists, axis=1))
                selected_subgoal = self.subgoals[selected_ind:selected_ind + 1]
                selected_subgoals = np.concatenate((selected_subgoals, selected_subgoal), axis=0)

                # delete the newly selected goal
                self.subgoals = np.delete(self.subgoals, selected_ind, axis=0)
                dists = np.delete(dists, selected_ind, axis=0)

                # update distance
                new_dist = np.linalg.norm(
                    np.expand_dims(selected_subgoal, axis=0) - np.expand_dims(self.subgoals, axis=1),
                    axis=-1)

                dists = np.concatenate((dists, new_dist), axis=1)

            self.subgoals = selected_subgoals

    def update_policy(self):
        self.n_traj = 0
        self.sample_batch()
        self.split_episode()

        self.generate_subgoals()
        self.reset_training_data()
        if self.sampled_goal_num is None or self.sampled_goal_num > 0:
            self.generate_fake_data()

        if self.norm_ob:
            self.ob_rms['observation'].update(self.state.cpu().numpy())
            self.ob_rms['desired_goal'].update(self.goal.cpu().numpy())

        if self.norm_ob:
            self.state = torch.clamp((self.state - torch.Tensor(self.ob_mean).type_as(self.state).unsqueeze(0)) /
                                     torch.sqrt(
                                         torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(self.state).unsqueeze(
                                             0)),
                                     -5, 5)
            self.goal = torch.clamp((self.goal - torch.Tensor(self.goal_mean).type_as(self.state).unsqueeze(0)) /
                                    torch.sqrt(torch.clamp(torch.Tensor(self.goal_var), 1e-4).type_as(
                                        self.state).unsqueeze(0)),
                                    -5, 5)
        if self.norm_rw:
            self.ret_rms.update(self.ret.squeeze(1).cpu().numpy())
            self.reward = torch.clamp(self.reward /
                                      torch.sqrt(torch.clamp(torch.Tensor([self.rw_var, ]), 1e-8)).type_as(self.state),
                                      -10., 10.)

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

        self.L = - (likelihood_ratio * self.A).sum() / self.n_traj

        self.policy.zero_grad()
        self.L.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer_pi.step()
        self.learn_step_counter += 1
        self.cur_kl = self.mean_kl_divergence().item()

        if self.norm_ob:
            self.ob_mean = self.ob_rms['observation'].mean
            self.ob_var = self.ob_rms['observation'].var
            self.goal_mean = self.ob_rms['desired_goal'].mean
            self.goal_var = self.ob_rms['desired_goal'].var
        if self.norm_rw:
            self.rw_var = self.ret_rms.var

    def reset_training_data(self):
        self.state = torch.Tensor(size=(0,) + self.state.size()[1:]).type_as(self.state)
        self.next_state = torch.Tensor(size=(0,) + self.next_state.size()[1:]).type_as(self.next_state)
        self.action = torch.Tensor(size=(0,) + self.action.size()[1:]).type_as(self.action)
        self.reward = torch.Tensor(size=(0,) + self.reward.size()[1:]).type_as(self.reward)
        self.ret = torch.Tensor(size=(0,) + self.reward.size()[1:]).type_as(self.reward)
        self.done = torch.Tensor(size=(0,) + self.done.size()[1:]).type_as(self.done)
        self.goal = torch.Tensor(size=(0,) + self.goal.size()[1:]).type_as(self.goal)
        self.gamma_discount = torch.Tensor(size=(0,) + self.gamma_discount.size()[1:]).type_as(self.gamma_discount)
        self.hratio = torch.Tensor(size=(0,) + self.hratio.size()[1:]).type_as(self.hratio)
        self.logpi_old = torch.Tensor(size=(0,) + self.logpi_old.size()[1:]).type_as(self.logpi_old)
        if self.discrete_action:
            self.distribution = torch.Tensor(size=(0,) + self.distribution.size()[1:]).type_as(self.distribution)
        else:
            self.mu = torch.Tensor(size=(0,) + self.mu.size()[1:]).type_as(self.mu)
            self.sigma = torch.Tensor(size=(0,) + self.sigma.size()[1:]).type_as(self.sigma)
        self.achieved_goal = torch.Tensor(size=(0,) + self.achieved_goal.size()[1:]).type_as(self.achieved_goal)
        self.desired_goal = torch.Tensor(size=(0,) + self.desired_goal.size()[1:]).type_as(self.desired_goal)
        self.n_traj = 0

    def split_episode(self, **kwargs):
        self.num_valid_ep = 0
        # Episodes store all the original data and will be used to generate fake
        # data instead of being used to train model
        assert self.other_data, "Hindsight algorithms need goal infos."
        self.desired_goal = self.other_data['desired_goal']
        self.achieved_goal = self.other_data['achieved_goal']
        self.goal = self.desired_goal

        self.episodes = []
        endpoints = (0,) + tuple((torch.nonzero(self.done[:, 0] > 0) + 1).squeeze().cpu().numpy().tolist())

        self.reward += 1
        suc_poses = torch.nonzero(self.reward == 1)[:, 0]
        for suc_pos in suc_poses:
            temp = suc_pos - torch.Tensor(endpoints).type_as(self.reward) + 1
            temp = temp[temp > 0]
            self.reward[suc_pos] = self.max_steps - torch.min(temp) + 1

        for i in range(len(endpoints) - 1):
            #
            self.num_valid_ep += \
                np.unique(np.round(self.achieved_goal[endpoints[i]: endpoints[i + 1]].cpu().numpy(), decimals=2),
                          axis=0).shape[0] > 1

            episode = {'state': self.state[endpoints[i]: endpoints[i + 1]],
                       'action': self.action[endpoints[i]: endpoints[i + 1]],
                       'reward': self.reward[endpoints[i]: endpoints[i + 1]],
                       'ret': torch.Tensor(self.reward[endpoints[i]: endpoints[i + 1]].shape).
                           zero_().type_as(self.reward[endpoints[i]: endpoints[i + 1]]),
                       'done': self.done[endpoints[i]: endpoints[i + 1]],
                       'next_state': self.next_state[endpoints[i]: endpoints[i + 1]],
                       'logpi_old': self.logpi_old[endpoints[i]: endpoints[i + 1]],
                       'desired_goal': self.desired_goal[endpoints[i]: endpoints[i + 1]],
                       'achieved_goal': self.achieved_goal[endpoints[i]: endpoints[i + 1]],
                       'length': endpoints[i + 1] - endpoints[i],
                       'gamma_discount': torch.Tensor(np.arange(endpoints[i + 1] - endpoints[i])).type_as(
                           self.state).unsqueeze(1)
                       }
            episode['ret'][-1, :] = episode['reward'][-1, :]
            for t in range(episode['reward'].shape[0] - 2, -1, -1):
                episode['ret'][t, :] = episode['reward'][t, :] + self.gamma * episode['ret'][t + 1, :]

            if self.discrete_action:
                episode['distribution'] = self.distribution[endpoints[i]: endpoints[i + 1]]
            else:
                episode['mu'] = self.mu[endpoints[i]: endpoints[i + 1]]
                episode['sigma'] = self.sigma[endpoints[i]: endpoints[i + 1]]

            self.episodes.append(episode)

        self.ret = torch.cat([ep['ret'] for ep in self.episodes], dim=0)
        self.gamma_discount = torch.cat([ep['gamma_discount'].squeeze(1) for ep in self.episodes], dim=0)
        self.n_traj += len(self.episodes)
        self.hratio = torch.ones(self.state.size(0)).type_as(self.state)

    def estimate_value(self):
        if self.value_func_type is not None:
            VPG.estimate_value(self)
        else:
            self.estimate_value_with_mc()

    def estimate_value_with_mc(self):
        # initialize a 1-d tensor to store returns
        returns = torch.zeros(self.reward.size(0), 1).type_as(self.state)
        hratio = self.hratio.unsqueeze(1)
        h_reward = self.reward * hratio

        # initialize 1-d vectors to store starting points and ending points of episodes
        end_points = torch.nonzero(self.done.squeeze() == 1).squeeze()
        # starting points are actually the ending points of the last episode, used only for computing epi_length
        start_points = - torch.ones(size=end_points.size()).type_as(end_points)
        start_points[1:] = end_points[:-1]
        # compute episode lengths
        ep_lens = end_points - start_points
        assert ep_lens.min().item() > 0, "Some episode lengths are smaller than 0."
        max_len = torch.max(ep_lens).item()
        # TODO: What's incomplete_flag
        incomplete_flag = ep_lens > 0

        # directly estimate return with h_reward
        returns[end_points] = h_reward[end_points]
        for i in range(1, max_len):
            incomplete_flag[ep_lens <= i] = 0
            indexes = (end_points - i)[incomplete_flag]
            returns[indexes] = self.reward[indexes] + self.gamma * returns[indexes + 1]

        self.R = returns.squeeze().detach()
        self.A = self.R
