import os
import abc
import copy
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from agents.Agent import Agent
from agents.config import VPG_CONFIG
from basenets.base_PG import FC_VALUE_FUNC


########################
# Define Value Network #
########################

class VPG(Agent):
    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        config = copy.deepcopy(VPG_CONFIG)
        config.update(parameters)
        super(VPG, self).__init__(config)
        self.max_grad_norm = config['max_grad_norm']    # TODO:What is max_grad_norm?
        self.steps_per_iter = config['steps_per_iter']
        # TODO:self.entropy_weight = config['entropy_weight']
        self.value_func_type = config['value_func_type']
        self.policy_type = config['policy_type']
        self.using_KL_estimation = config['using_KL_estimation']
        self.learn_step_counter = 0

        # initialize tensor and scalar for computed loss
        self.loss_v = None
        self.value_loss = 0.
        self.L = 0.
        self.n_traj = 0
        # TODO:self.policy_ent = 0.

        # initialize variables to store values, advantages and estimated returns
        self.V = None
        self.A = None
        self.R = None
        self.estimated_R = None

        # initialize variables used in following functions
        self.num_valid_ep = 0
        self.desired_goal = None
        self.achieved_goal = None
        self.goal = None
        self.episodes = None
        self.gamma_discount = None
        self.distribution = None
        self.mu = None
        self.sigma = None
        self.sample_index = None
        self.optimizer_pi = None

        # initialize value function estimator / value network
        if self.value_func_type is not None:
            # initialize value network architecture
            if self.value_func_type == 'FC':
                self.value_func = FC_VALUE_FUNC(
                    self.num_states['state_dim'] + self.num_states['goal_dim'] \
                                if isinstance(self.num_states, dict) else self.num_states,
                    num_hiddens=config['hidden_layers_v'] if isinstance(config['hidden_layers_v'], list)
                                                        else config['hidden_layers'],
                    batch_normalization=self.batch_normalization,
                    activ_func=self.activ_func,
                )
            elif self.value_func_type == 'Conv':
                raise NotImplementedError("Not Implemented. Are there Conv value function estimators?")

            # initialize the type of loss function, learning rate, optimizer type and parameters for value function
            self.lamb = config['GAE_lambda']    # GAE-Lambda
            self.value_loss_func = config['value_loss_func'](reduction='mean')
            self.value_lr = config['value_lr']
            self.train_v_iters = config['train_v_iters']
            self.momentum_v = config['momentum_v']  # momentum: a parameter for optimizers
            if config['optimizer_v'] == optim.LBFGS:        # TODO: Why consider using LBFGS for optimization?
                self.using_lbfgs_for_V = True
            else:
                self.using_lbfgs_for_V = False
                if self.momentum_v is not None:
                    self.optimizer_v = config['optimizer_v'](self.value_func.parameters(), lr=self.value_lr,
                                                             momentum=self.momentum_v)
                else:
                    self.optimizer_v = config['optimizer_v'](self.value_func.parameters(), lr=self.value_lr)
        else:
            self.value_func = None
            raise AttributeError("Value Function is None.")

    # use cuda
    def cuda(self):
        Agent.cuda(self)
        self.policy = self.policy.cuda()
        if self.value_func_type is not None:
            self.value_func = self.value_func.cuda()
        else:
            raise AttributeError("Value Function Type is None.")

    def estimate_value(self):
        # initialize a 1-d tensor to store returns
        returns = torch.zeros(self.reward.size(0), 1).type_as(self.state)
        values = self.value_func(self.state, other_data=self.other_data)

        # record the index for early termination, those reached goals ahead of max_steps
        early_done = torch.nonzero(self.done.squeeze() == 2).squeeze(-1)

        # initialize a 1-d tensor to store TD-Errors
        delta = torch.zeros(self.reward.size(0), 1).type_as(self.state)

        # initialize a 1-d tensor to store advantages
        advantages = torch.zeros(self.reward.size(0), 1).type_as(self.state)

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

        # compute TD-Error, advantages and estimated return
        # TODO: Still don't know the following 3 lines
        delta[end_points] = self.reward[end_points] - values[end_points]
        if early_done.numel() > 0:
            delta[early_done] += self.value_func(self.next_state[early_done], other_data=self.other_data[early_done]
                                               if self.other_data is not None else None).resize_as(delta[early_done])
        advantages[end_points] = delta[end_points]
        returns[end_points] = self.reward[end_points]
        for i in range(1, max_len):
            incomplete_flag[ep_lens <= i] = 0
            indexes = (end_points - i)[incomplete_flag]
            delta[indexes] = self.reward[indexes] + self.gamma * values[indexes + 1] - values[indexes]
            advantages[indexes] = delta[indexes] + self.gamma * self.lamb * advantages[indexes + 1]
            returns[indexes] = self.reward[indexes] + self.gamma * returns[indexes + 1]

        estimated_return = values + advantages
        self.V = values.squeeze().detach()
        self.R = returns.squeeze().detach()
        self.A = advantages.squeeze().detach()
        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-10)
        self.estimated_R = estimated_return.squeeze().detach()

    def update_value(self, indexes=None):
        if indexes is None:
            indexes = np.arange(self.state.size(0))
        # Target value is the actual return, estimated value is the one computed by value function
        V_target = self.estimated_R[indexes]
        V_eval = self.value_func(self.state[indexes], other_data=self.other_data[indexes]
                                            if self.other_data is not None else None).squeeze()
        # tensor for computation, scalar for printing result
        self.loss_v = self.value_loss_func(V_eval, V_target)
        self.value_loss = self.loss_v.item()
        self.value_func.zero_grad()     # TODO: why zero_grad() here?
        self.loss_v.backward()
        # TODO: some sort of operation?
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.value_func.parameters(), self.max_grad_norm)
        self.optimizer_v.step()

    @abc.abstractmethod
    def compute_ratio(self, indexes=None, curr_policy=None):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def mean_kl_divergence(self, inds=None, model=None):
        raise NotImplementedError("Must be implemented in subclass.")

    def sample_batch(self, batch_size=None):
        batch, self.sample_index = Agent.sample_batch(self, batch_size=None)
        self.reward = self.reward.resize_(batch['reward'].shape).copy_(torch.Tensor(batch['reward']))
        self.done = self.done.resize_(batch['done'].shape).copy_(torch.Tensor(batch['done']))
        self.action = self.action.resize_(batch['action'].shape).copy_(torch.Tensor(batch['action']))
        self.state = self.state.resize_(batch['state'].shape).copy_(torch.Tensor(batch['state']))
        self.next_state = \
            self.next_state.resize_(batch['next_state'].shape).copy_(torch.Tensor(batch['next_state']))
        self.logpi_old = self.logpi_old.resize_(batch['logpi'].shape).copy_(torch.Tensor(batch['logpi']))
        if self.discrete_action:
            self.distribution = \
                self.distribution.resize_(batch['distribution'].shape).copy_(torch.Tensor(batch['distribution']))
        else:
            self.mu = self.mu.resize_(batch['mu'].shape).copy_(torch.Tensor(batch['mu']))
            self.sigma = self.sigma.resize_(batch['sigma'].shape).copy_(torch.Tensor(batch['sigma']))
        self.other_data = batch['other_data']
        if self.other_data:
            for key in self.other_data.keys():
                self.other_data[key] = torch.Tensor(self.other_data[key]).type_as(self.state)

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
        suc_poses = torch.nonzero(self.reward==1)[:, 0]
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
                       'done': self.done[endpoints[i]: endpoints[i + 1]],
                       'next_state': self.next_state[endpoints[i]: endpoints[i + 1]],
                       'logpi_old': self.logpi_old[endpoints[i]: endpoints[i + 1]],
                       'desired_goal': self.desired_goal[endpoints[i]: endpoints[i + 1]],
                       'achieved_goal': self.achieved_goal[endpoints[i]: endpoints[i + 1]],
                       'length': endpoints[i + 1] - endpoints[i],
                       'gamma_discount': torch.Tensor(np.arange(endpoints[i + 1] - endpoints[i])).type_as(
                                            self.state).unsqueeze(1)
                       }

            if self.discrete_action:
                episode['distribution'] = self.distribution[endpoints[i]: endpoints[i + 1]]
            else:
                episode['mu'] = self.mu[endpoints[i]: endpoints[i + 1]]
                episode['sigma'] = self.sigma[endpoints[i]: endpoints[i + 1]]

            self.episodes.append(episode)

        self.gamma_discount = torch.cat([ep['gamma_discount'].squeeze(1) for ep in self.episodes], dim=0)
        self.n_traj += len(self.episodes)

    def update_policy(self):
        self.n_traj = 0  # refer to learn_trpo
        self.sample_batch()
        self.split_episode()
        self.other_data = self.goal     # refer to learn_trpo
        self.estimate_value()

        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-8)  # TODO: Why uniforming advantages?
        likelihood_ratio = self.compute_ratio()
        self.L = - (likelihood_ratio * self.A).mean()   # TODO: I changed self.A.squeeze() to self.A
        if self.value_func_type is not None:
            # update value for train_v_iters times
            for i in range(self.train_v_iters):
                self.update_value()
        self.policy.zero_grad()
        self.L.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer_pi.step()
        self.learn_step_counter += 1

    def save_model(self, save_path):
        print("saving models...")
        save_dict = {
            'policy': self.policy.state_dict(),
            'optimizer_pi': self.optimizer_pi.state_dict(),
            'episode': self.episode_counter,
            'step': self.learn_step_counter,
        }
        torch.save(save_dict, os.path.join(save_path, "policy" + str(self.learn_step_counter) + ".pth"))
        if self.value_func_type is not None and not self.using_lbfgs_for_V:
            save_dict = {
                'policy': self.value_func.state_dict(),
                'optimizer_pi': self.optimizer_v.state_dict(),
            }
            torch.save(save_dict, os.path.join(save_path, "value" + str(self.learn_step_counter) + ".pth"))

    def load_model(self, load_path, load_point):
        policy_name = os.path.join(load_path, "policy" + str(load_point) + ".pth")
        print("loading checkpoint %s" % policy_name)
        checkpoint = torch.load(policy_name)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer_pi.load_state_dict(checkpoint['optimizer_pi'])
        self.learn_step_counter = checkpoint['step']
        self.episode_counter = checkpoint['episode']
        print("loaded checkpoint %s" % policy_name)
