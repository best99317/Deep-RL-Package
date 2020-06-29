import torch
import numpy as np
from agents.HPG.HPG import HPG
from agents.VPG.VPG_Softmax import VPG_Softmax


class HPG_Softmax(HPG, VPG_Softmax):
    def __init__(self, parameters):
        super(HPG_Softmax, self).__init__(parameters)

    def generate_fake_data(self):
        self.subgoals = torch.Tensor(self.subgoals).type_as(self.state)
        # number of subgoals
        n_g = self.subgoals.shape[0]

        # for weighted importance sampling, Ne x Ng x T
        # h_ratios initialized to replace the original ones
        h_ratios = torch.zeros(size=(len(self.episodes), n_g, self.max_steps)).type_as(self.state)
        h_ratios_mask = torch.zeros(size=(len(self.episodes), n_g, self.max_steps)).type_as(self.state)

        # copy the data in episodes to fake reward, length and dones according to hindsight methodology
        for ep in range(len(self.episodes)):
            # original episode length
            ep_len = self.episodes[ep]['length']

            # Modify episode length and rewards.
            # Ng x T
            # Turn the reward of the achieved goals to 1
            reward_fake = self.env.compute_reward(
                self.episodes[ep]['achieved_goal'].unsqueeze(0).repeat(n_g, 1, 1).cpu().numpy(),
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

            reward_fake = torch.Tensor(reward_fake).type_as(self.reward)

            # Rewards are 0 and T - t_done + 1
            # Turn the reward of the trajectories to achieved goals to T - t_done + 1
            # Ng x T
            reward_fake[range(reward_fake.size(0)), length_fake - 1] = \
                (self.max_steps - length_fake + 1).type_as(self.reward)
            reward_fake[neg_ep_inds] = 0

            dones_fake = self.episodes[ep]['done'].squeeze().repeat(n_g, 1)
            dones_fake[pos_ep_inds, length_fake[pos_ep_inds] - 1] = 1

            # Ng x T
            mask = torch.Tensor(np.arange(1, ep_len + 1)).type_as(self.state).repeat(n_g, 1)
            mask[mask > length_fake.type_as(self.state).unsqueeze(1)] = 0
            mask[mask > 0] = 1
            # filter out the episodes where at beginning, the goal is achieved.
            mask[length_fake == 1] = 0

            h_ratios_mask[ep][:, :ep_len] = mask

            # in this case, the input state is the full state of the envs, which should be a vector.
            if self.policy_type == 'FC':
                expanded_s = self.episodes[ep]['state'][:ep_len].repeat(n_g, 1)
            # in this case, the input state is represented by images
            elif self.episodes[ep]['state'].dim() == 4:
                expanded_s = self.episodes[ep]['state'][:ep_len].repeat(n_g, 1, 1, 1)
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

            fake_distribution = self.policy(fake_input_state, other_data=fake_input_goal).detach()

            # Ng * T x Da
            expanded_a = self.episodes[ep]['action'].repeat(n_g, 1)

            # Ng x T
            fake_logpi = self.compute_logp(fake_distribution, expanded_a).reshape(n_g, ep_len)
            expanded_logpi_old = self.episodes[ep]['logpi_old'].repeat(n_g, 1).reshape(n_g, -1)
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
            self.next_state = torch.cat((self.next_state, self.episodes[ep]['next_state'].repeat(n_g, 1)[mask]), dim=0)
            self.action = torch.cat((self.action, expanded_a[mask]), dim=0)

            self.goal = torch.cat((self.goal, expanded_g[mask]), dim=0)
            self.distribution = torch.cat((self.distribution, fake_distribution[mask]), dim=0)

            self.reward = torch.cat((self.reward, reward_fake.reshape(n_g * ep_len, 1)[mask]), dim=0)
            self.done = torch.cat((self.done, dones_fake.reshape(n_g * ep_len, 1)[mask]), dim=0)
            self.logpi_old = torch.cat((self.logpi_old, fake_logpi.reshape(n_g * ep_len, 1)[mask]), dim=0)

            # Ng x T
            gamma_discount = torch.pow(self.gamma, torch.Tensor(np.arange(1, ep_len + 1)).type_as(self.state)).repeat(n_g,
                                                                                                                      1)
            self.gamma_discount = torch.cat((self.gamma_discount, gamma_discount.reshape(n_g * ep_len)[mask]), dim=0)

            self.n_traj += n_g

        if self.weight_is:
            h_ratios_sum = torch.sum(h_ratios, dim=0, keepdim=True)
            h_ratios /= h_ratios_sum

        h_ratios_mask = h_ratios_mask.reshape(-1) > 0
        self.hratio = torch.cat((self.hratio, h_ratios.reshape(-1)[h_ratios_mask]), dim=0)

    def choose_action(self, s, other_data = None, greedy = False):
        assert other_data is None or other_data.size(-1) == self.dim_goal, "other_data should only contain goal information in current version"
        # TODO: Without the following content, the algorithm would not converge at all...
        if self.norm_ob:
            s = torch.clamp((s - torch.Tensor(self.ob_mean).type_as(s).unsqueeze(0)) / torch.sqrt(
                torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(s).unsqueeze(0)), -5, 5)
            other_data = torch.clamp((other_data - torch.Tensor(self.goal_mean).type_as(s).unsqueeze(0)) / torch.sqrt(
                torch.clamp(torch.Tensor(self.goal_var), 1e-4).type_as(s).unsqueeze(0)), -5, 5)
        return VPG_Softmax.choose_action(self, s, other_data, greedy)
