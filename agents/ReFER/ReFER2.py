import numpy as np
import torch
import copy
from agents.HTRPO.HTRPO import HTRPO
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector


class ReFER2(HTRPO):

    def __init__(self, parameters):
        super().__init__(parameters)
        self.C = 2
        self.N = 800
        self.D = 0.1
        self.B = 5000
        self.A_coef = 5e-7
        self.eta = 0.0001
        self.c_max = 1 + self.C / (1 + self.A_coef * self.learn_step_counter * self.steps_per_iter)
        self.ptr = 0
        # self.hratio_unweighted = 0
        self.success_buffer = []

    def reset_training_data(self):
        self.state = torch.Tensor(size=(0,) + self.state.size()[1:]).type_as(self.state)
        self.next_state = torch.Tensor(size=(0,) + self.next_state.size()[1:]).type_as(self.next_state)
        self.action = torch.Tensor(size=(0,) + self.action.size()[1:]).type_as(self.action)
        self.reward = torch.Tensor(size=(0,) + self.reward.size()[1:]).type_as(self.reward)
        self.done = torch.Tensor(size=(0,) + self.done.size()[1:]).type_as(self.done)
        self.goal = torch.Tensor(size=(0,) + self.goal.size()[1:]).type_as(self.goal)
        self.gamma_discount = torch.Tensor(size=(0,) + self.gamma_discount.size()[1:]).type_as(self.gamma_discount)
        self.hratio = torch.Tensor(size=(0,) + self.hratio.size()[1:]).type_as(self.hratio)
        # self.hratio_unweighted = \
        #     torch.Tensor(size=(0,) + self.hratio_unweighted.size()[1:]).type_as(self.hratio_unweighted)
        self.logpi_old = torch.Tensor(size=(0,) + self.logpi_old.size()[1:]).type_as(self.logpi_old)
        if self.discrete_action:
            self.distribution = torch.Tensor(size=(0,) + self.distribution.size()[1:]).type_as(self.distribution)
        else:
            self.mu = torch.Tensor(size=(0,) + self.mu.size()[1:]).type_as(self.mu)
            self.sigma = torch.Tensor(size=(0,) + self.sigma.size()[1:]).type_as(self.sigma)
        self.achieved_goal = torch.Tensor(size=(0,) + self.achieved_goal.size()[1:]).type_as(self.achieved_goal)
        self.desired_goal = torch.Tensor(size=(0,) + self.desired_goal.size()[1:]).type_as(self.desired_goal)
        self.n_traj = 0

    # def get_near_policy(self):
    #     self.c_max = 1 + self.C / (1 + self.A_coef * self.learn_step_counter * self.steps_per_iter)
    #     indexes = self.hratio_unweighted.lt(self.c_max) & self.hratio_unweighted.gt(1. / self.c_max)
    #     self.state = self.state[indexes == 1]
    #     self.next_state = self.next_state[indexes == 1]
    #     self.action = self.action[indexes == 1]
    #     self.reward = self.reward[indexes == 1]
    #     self.done = self.done[indexes == 1]
    #     self.goal = self.goal[indexes == 1]
    #     self.hratio = self.hratio[indexes == 1]
    #     self.gamma_discount = self.gamma_discount[indexes == 1]
    #     self.logpi_old = self.logpi_old[indexes == 1]
    #     if self.discrete_action:
    #         self.distribution = self.distribution[indexes == 1]
    #     else:
    #         self.mu = self.mu[indexes == 1]
    #         self.sigma = self.sigma[indexes == 1]

    def store_successful_data(self):
        # check the length of the episode, if the last reward is not 0 and length longer than 1
        for episode in self.episodes:
            if episode['reward'][-1] != 0 and episode['length'] != 1:
                self.success_buffer.append(episode)
                self.ptr += episode['length']
                # no need to compute A, for it'll be done in parallel
        # control the length of success_buffer can be added.
        if self.ptr >= self.B:
            pop_data = self.ptr - self.B
            if pop_data:
                while pop_data != 0:
                    if self.success_buffer[0]['length'] <= pop_data:
                        del self.success_buffer[0]
                        pop_data -= self.success_buffer[0]['length']
                        self.ptr -= self.success_buffer[0]['length']
        print("Number of History Trajectories: ", len(self.success_buffer),
              " | Number of History Data: ", self.ptr)
                    # else:
                    #     for key in self.success_buffer[0]:
                    #         if key == 'length':
                    #             self.success_buffer[0][key] -= pop_data
                    #         else:
                    #             self.success_buffer[0][key] = self.success_buffer[0][key][pop_data:]
                    #     self.ptr -= pop_data
                    #     pop_data = 0



    def get_successful_data(self):
        for episode in self.success_buffer:
            self.state = torch.cat((self.state, episode['state']), dim=0)
            self.next_state = torch.cat((self.next_state, episode['next_state']), dim=0)
            self.action = torch.cat((self.action, episode['action']), dim=0)

            self.goal = torch.cat((self.goal, episode['desired_goal']), dim=0)
            self.mu = torch.cat((self.mu, episode['mu']), dim=0)
            self.sigma = torch.cat((self.sigma, episode['sigma']), dim=0)

            self.reward = torch.cat((self.reward, episode['reward']), dim=0)
            self.done = torch.cat((self.done, episode['done']), dim=0)
            self.logpi_old = torch.cat((self.logpi_old, episode['logpi_old']), dim=0)

            gamma_discount = torch.pow(self.gamma, torch.Tensor(np.arange(1, episode['length'] + 1)).type_as(self.state))
            self.gamma_discount = torch.cat((self.gamma_discount, gamma_discount), dim=0)
            self.n_traj += 1
            self.hratio = torch.cat((self.hratio, torch.ones(episode['length']).cuda()), dim=0)



    def generate_fake_data(self):
        self.subgoals = torch.Tensor(self.subgoals).type_as(self.state)
        # need to get the fake data for multiple past iterations
        self.get_fake_data_per_epi(self.episodes)

    def get_fake_data_per_epi(self, episodes):
        # number of subgoals
        n_g = self.subgoals.shape[0]

        h_ratios = torch.zeros(size=(len(episodes), n_g, self.max_steps)).type_as(self.state)
        h_ratios_mask = torch.zeros(size=(len(episodes), n_g, self.max_steps)).type_as(self.state)

        # copy the data in episodes to fake reward, length and dones according to hindsight methodology
        for ep in range(len(episodes)):
            # original episode length
            ep_len = episodes[ep]['length']

            # Modify episode length and rewards.
            # Ng x T
            # Turn the reward of the achieved goals to 1
            reward_fake = self.env.compute_reward(
                episodes[ep]['achieved_goal'].unsqueeze(0).repeat(n_g, 1, 1).cpu().numpy(),
                self.subgoals.unsqueeze(1).repeat(1, ep_len, 1).cpu().numpy(), None)
            # Here, reward will be 0 when the goal is not achieved, else 1.
            reward_fake += 1
            # For negative episode, there is no positive reward, all are 0.
            neg_ep_inds = np.where(reward_fake.sum(axis=-1) == 0)
            pos_ep_inds = np.where(reward_fake.sum(axis=-1) > 0)

            # In reward, there are only 0 and 1. The first 1's position indicates the episode length.
            length_fake = np.argmax(reward_fake, axis=-1)
            length_fake += 1
            # For all negative episodes, the length is the value of max_steps.
            length_fake[neg_ep_inds] = ep_len
            # lengths: Ng
            length_fake = torch.Tensor(length_fake).type_as(self.state).long()

            # Ng x T
            mask = torch.Tensor(np.arange(1, ep_len + 1)).type_as(self.state).repeat(n_g, 1)
            mask[mask > length_fake.type_as(self.state).unsqueeze(1)] = 0
            mask[mask > 0] = 1
            # filter out the episodes where at beginning, the goal is achieved.
            mask[length_fake == 1] = 0

            reward_fake = torch.Tensor(reward_fake).type_as(self.reward)

            # Rewards are 0 and T - t_done + 1
            # Turn the reward of the trajectories to achieved goals to T - t_done + 1
            # Ng x T
            reward_fake[range(reward_fake.size(0)), length_fake - 1] = \
                (self.max_steps - length_fake + 1).type_as(self.reward)
            reward_fake[neg_ep_inds] = 0

            dones_fake = episodes[ep]['done'].squeeze().repeat(n_g, 1)
            dones_fake[pos_ep_inds, length_fake[pos_ep_inds] - 1] = 1

            h_ratios_mask[ep][:, :ep_len] = mask

            # in this case, the input state is the full state of the envs, which should be a vector.
            if self.policy_type == 'FC':
                expanded_s = episodes[ep]['state'][:ep_len].repeat(n_g, 1)
            # in this case, the input state is represented by images
            elif episodes[ep]['state'].dim() == 4:
                expanded_s = episodes[ep]['state'][:ep_len].repeat(n_g, 1, 1, 1)
            else:
                expanded_s = None
                raise NotImplementedError

            expanded_g = self.subgoals.unsqueeze(1).repeat(1, ep_len, 1).reshape(-1, self.dim_goal)
            # - self.episodes[ep]['achieved_goal'].unsqueeze(0).repeat(n_g,1,1).reshape(-1, self.d_goal)
            if self.norm_ob:
                fake_input_state = torch.clamp(
                    (expanded_s - torch.Tensor(self.ob_mean).type_as(self.state).unsqueeze(0)) / torch.sqrt(
                        torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(self.state).unsqueeze(0)), -5, 5)
                fake_input_goal = torch.clamp(
                    (expanded_g - torch.Tensor(self.goal_mean).type_as(self.state).unsqueeze(0)) / torch.sqrt(
                        torch.clamp(torch.Tensor(self.goal_var), 1e-4).type_as(self.state).unsqueeze(0)), -5, 5)

            else:
                fake_input_state = expanded_s
                fake_input_goal = expanded_g

            fake_mu, fake_log_sigma, fake_sigma = self.policy(fake_input_state, other_data=fake_input_goal)
            fake_mu = fake_mu.detach()
            fake_sigma = fake_sigma.detach()
            fake_log_sigma = fake_log_sigma.detach()

            # Ng * T x Da
            expanded_a = episodes[ep]['action'].repeat(n_g, 1)

            # Ng x T
            fake_logpi = self.compute_logp(fake_mu, fake_log_sigma, fake_sigma, expanded_a).reshape(n_g, ep_len)
            expanded_logpi_old = episodes[ep]['logpi_old'].repeat(n_g, 1).reshape(n_g, -1)
            d_logp = fake_logpi - expanded_logpi_old

            # generate hindsight ratio
            # Ng x T
            if self.per_decision:
                h_ratio = torch.exp(d_logp.cumsum(dim=1)) + 1e-10
                h_ratio *= mask
                h_ratios[ep][:, :ep_len] = h_ratio
            else:
                h_ratio = torch.exp(torch.sum(d_logp, keepdim=True)).repeat(1, ep_len) + 1e-10
                h_ratio *= mask
                h_ratios[ep][:, :ep_len] = h_ratio

            # make all data one batch
            mask = mask.reshape(-1) > 0

            self.state = torch.cat((self.state, expanded_s[mask]), dim=0)
            self.next_state = torch.cat((self.next_state, episodes[ep]['next_state'].repeat(n_g, 1)[mask]), dim=0)
            self.action = torch.cat((self.action, expanded_a[mask]), dim=0)

            self.goal = torch.cat((self.goal, expanded_g[mask]), dim=0)
            self.mu = torch.cat((self.mu, fake_mu[mask]), dim=0)
            self.sigma = torch.cat((self.sigma, fake_sigma[mask]), dim=0)

            self.reward = torch.cat((self.reward, reward_fake.reshape(n_g * ep_len, 1)[mask]), dim=0)
            self.done = torch.cat((self.done, dones_fake.reshape(n_g * ep_len, 1)[mask]), dim=0)
            self.logpi_old = torch.cat((self.logpi_old, fake_logpi.reshape(n_g * ep_len, 1)[mask]), dim=0)

            # Ng x T
            gamma_discount = torch.pow(self.gamma, torch.Tensor(np.arange(1, ep_len + 1)).type_as(self.state)).repeat(n_g,
                                                                                                                      1)
            self.gamma_discount = torch.cat((self.gamma_discount, gamma_discount.reshape(n_g * ep_len)[mask]), dim=0)

            self.n_traj += n_g

        if self.weight_is:
            # hratio_unweighted = copy.deepcopy(h_ratios)
            h_ratios_sum = torch.sum(h_ratios, dim=0, keepdim=True)
            h_ratios /= h_ratios_sum

        h_ratios_mask = h_ratios_mask.reshape(-1) > 0
        self.hratio = torch.cat((self.hratio, h_ratios.reshape(-1)[h_ratios_mask]), dim=0)
        print("Number of Hindsight Trajectories: ", self.n_traj, " | Number of Hindsight Data: ", len(self.state))
        # self.hratio_unweighted = \
        #     torch.cat((self.hratio_unweighted, hratio_unweighted.reshape(-1)[h_ratios_mask]), dim=0)

    def split_episode(self, **kwargs):
        # do the general job as HTRPO does
        self.num_valid_ep = 0
        # Episodes store all the original data and will be used to generate fake
        # data instead of being used to train model
        assert self.other_data, "Hindsight algorithms need goal infos."
        self.desired_goal = self.other_data['desired_goal']
        self.achieved_goal = self.other_data['achieved_goal']
        self.goal = self.desired_goal
        # initialize real data's hindsight ratios.
        self.hratio = torch.ones(self.state.size(0)).type_as(self.state)
        # self.hratio_unweighted = torch.ones(self.state.size(0)).type_as(self.state)
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

        # done splitting the episodes of this batch, then  add this-batch episodes to episodes buffer
        # TODO: should define a function to control the length of episodes buffer?
        # TODO: don't know if numpy works with list of dictionaries?
        # if self.ptr >= self.B:
        #     pop_episodes = (self.ptr % self.B) // self.steps_per_iter
        #     pop_data = self.ptr % self.B - pop_episodes * self.steps_per_iter
        #     if pop_episodes:
        #         del self.episodes_buf[0:pop_episodes]
        #         self.ptr -= pop_episodes * self.steps_per_iter
        #     if pop_data:
        #         while pop_data != 0:
        #             if self.episodes_buf[0][0]['length'] <= pop_data:
        #                 del self.episodes_buf[0][0]
        #                 pop_data -= self.episodes_buf[0][0]['length']
        #                 self.ptr -= self.episodes_buf[0][0]['length']
        #             else:
        #                 for key in self.episodes_buf[0][0]:
        #                     if key == 'length':
        #                         self.episodes_buf[0][0][key] -= pop_data
        #                     else:
        #                         self.episodes_buf[0][0][key] = self.episodes_buf[0][0][key][pop_data:]
        #                 self.ptr -= pop_data
        #                 pop_data = 0
        # self.episodes_buf.append(self.episodes)

    def sample_batch(self, batch_size=32):
        # sample the data of this batch
        HTRPO.sample_batch(self)
        # self.ptr += self.steps_per_iter
        # split_episode, and then add the episodes of this batch to episodes buffer
        # after this, the episodes of past iterations are stored
        self.split_episode()

        # generate subgoals of this batch
        # TODO: or generate subgoals for all batches?
        self.generate_subgoals()

        # clear up self.state etc. to be filled with actual training data
        self.reset_training_data()

        # should generate fake data for all chosen past real data
        if self.sampled_goal_num is None or self.sampled_goal_num > 0:
            self.generate_fake_data()

        self.get_successful_data()
        self.store_successful_data()

        if self.norm_ob:
            self.ob_rms['observation'].update(self.state.cpu().numpy())
            self.ob_rms['desired_goal'].update(self.goal.cpu().numpy())

        # self.get_near_policy()

    def update_policy(self):
        self.n_traj = 0

        # direcly get the final sampled near-policy fake data used for training
        self.sample_batch()

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