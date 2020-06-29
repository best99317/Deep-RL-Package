import torch
import abc
import copy
from .config import AGENT_CONFIG
from collections import deque


class Agent:
    __metaclass__ = abc.ABCMeta

    def __init__(self, parameters):
        config = copy.deepcopy(AGENT_CONFIG)
        config.update(parameters)
        self.num_states = config['num_states']

        if 'num_actions' in config.keys():
            self.num_actions = config['num_actions']
        self.action_dims = config['action_dims']
        self.lr_pi = config['lr_pi']
        self.momentum = config['momentum']
        self.gamma = config['reward_decay']
        self.memory_size = config['memory_size']
        self.hidden_layers = config['hidden_layers']
        self.activ_func = config['activ_func']
        self.out_activ_func = config['out_activ_func']
        self.discrete_action = config['discrete_action']
        self.batch_normalization = config['batch_normalization']
        self.action_dims = config['action_dims']
        self.memory = None
        self.policy = None
        self.value_func = None
        # TODO:self.norm_ob = None

        self.step_counter = 0
        self.episode_counter = 0

        # used in HTRPO. In other algorithms, it will be set to 0.
        self.max_steps = 0

        # TODO:self.cost_his = []

        self.use_cuda = False
        self.reward = torch.Tensor(1)
        self.done = torch.Tensor(1)
        self.next_state = torch.Tensor(1)
        self.state = torch.Tensor(1)
        self.action = torch.Tensor(1)
        self.logpi_old = torch.Tensor(1)
        self.other_data = None

    @abc.abstractmethod
    def choose_action(self, state, other_data=None, greedy=False):
        raise NotImplementedError("Must be implemented in subclass.")

    # @abc.abstractmethod
    # def learn(self):
    #     raise NotImplementedError("Must be implemented in subclass.")

    def store_transition(self, transition):
        self.memory.store_transition(transition)

    def sample_batch(self, batch_size=None):
        return self.memory.sample_batch(batch_size)

    #
    # def soft_update(self, target, eval, tau):
    #     for target_param, param in zip(target.parameters(), eval.parameters()):
    #         target_param.data.copy_(target_param.data * (1.0 - tau) +
    #                                 param.data * tau)
    # 
    # def hard_update(self, target, eval):
    #     target.load_state_dict(eval.state_dict())
    #     # print('\ntarget_params_replaced\n')

    def cuda(self):
        self.use_cuda = True
        self.reward = self.reward.cuda()
        # TODO: Unknown variables
        self.action = self.action.cuda()
        self.state = self.state.cuda()
        self.next_state = self.next_state.cuda()
        self.done = self.done.cuda()
        self.logpi_old = self.logpi_old.cuda()

    @abc.abstractmethod
    def save_model(self, save_path):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def load_model(self, load_path, load_point):
        raise NotImplementedError("Must be implemented in subclass.")

    def eval_policy(self, env, render=False, eval_num=None):
        ep_rew_list = deque(maxlen=eval_num)
        # success_history = deque(maxlen=eval_num)
        self.policy = self.policy.eval()
        if self.value_func is not None:
            self.value_func = self.value_func.eval()
        observation = env.reset()

        while len(ep_rew_list) < eval_num:

            for key in observation.keys():
                observation[key] = torch.Tensor(observation[key])

            if render:
                env.render()

            if isinstance(observation, dict):
                goal = observation["desired_goal"]
                observation = observation["observation"]
            else:
                goal = None

            if not self.discrete_action:
                actions, _, _, _ = self.choose_action(observation, other_data=goal, greedy=False)
            else:
                actions, _ = self.choose_action(observation, other_data=goal, greedy=False)
            actions = actions.cpu().numpy()
            observation, rewards, dones, infos = env.step(actions)
            for e, info in enumerate(infos):
                if dones[e]:
                    if 'reward' not in info['episode']:
                        info['episode']['reward'] = info['episode'].pop('r')
                        info['episode']['length'] = info['episode'].pop('l')
                    ep_rew_list.append(info.get('episode')['reward'] + self.max_steps)
                    # success_history.append(info.get('is_success'))
                    for k in observation.keys():
                        observation[k][e] = info.get('new_obs')[k]

        # return ep_rew_list, success_history
        return ep_rew_list,
