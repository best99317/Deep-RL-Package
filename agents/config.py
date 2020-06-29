from torch import optim
from torch.nn import MSELoss
import torch.nn.functional as F

AGENT_CONFIG = {
    # 'lr' indicates the learning rate of the "mainbody".
    # Value-based RL: Q net; Policy-based RL: Policy net; Actor-critic RL: Actor
    'lr_pi':0.01,
    'momentum':None,
    'reward_decay':0.9,
    'memory_size': 1e6,
    # 'hidden_layers' defines the layers of the "mainbody".
    # Value-based RL: Q net; Policy-based RL: Policy net; Actor-critic RL: Actor
    'hidden_layers':[64, 64],
    'activ_func': F.tanh,
    'out_activ_func': None,
    'batch_normalization': False,
}

VPG_CONFIG = {
    'max_grad_norm': 2,
    'steps_per_iter': 2048,
    'action_bounds': 1,
    'optimizer_pi': optim.Adam,
    'optimizer_v': optim.Adam,
    'GAE_lambda': 0.95,
    'entropy_weight': 0.0,
    'init_noise': 1.,
    'value_func_type': 'FC',
    'hidden_layers_v' : [64,64],
    'value_loss_func': MSELoss,
    'momentum_v': None,
    'momentum_pi': None,
    'value_lr': 0.01,
    'train_v_iters': 3,
    'using_KL_estimation': False,
    'policy_type': 'FC',
}
VPG_CONFIG['memory_size'] = VPG_CONFIG['steps_per_iter']

NPG_CONFIG = {
    'cg_iters': 10,
    'cg_residual_tol' : 1e-10,
    'cg_damping': 1e-3,
    'max_kl_divergence':0.01,
}

TRPO_CONFIG = {
    'max_search_num' : 10,
    'accept_ratio' : .1,
    'step_frac': .5
}

HPG_CONFIG = {
    'sampled_goal_num': 10,
    'goal_space': None,
    'per_decision': True,
    'weighted_is': True,
}

HTRPO_CONFIG = {
    'hindsight_steps': 10,
    'sampled_goal_num': 10,
    'goal_space': None,
    'per_decision': True,
    'weighted_is': True,
    'using_hgf_goals' : True,
    'using_KL_estimation' : True,
}

ReFER_CONFIG = {
    'hindsight_steps': 10,
    'sampled_goal_num': 10,
    'goal_space': None,
    'per_decision': True,
    'weighted_is': True,
    'using_hgf_goals' : True,
    'using_KL_estimation' : True,
}

# NAF_CONFIG = {
#     'steps_per_iter': 50,
#     'learn_start_step': 10000,
#     'tau': 0.005,
#     'lr': 1e-3,
#     'noise_var': 0.3,
#     'noise_min': 0.01,
#     'noise_decrease': 2e-5,
#     'optimizer': optim.Adam,
#     'loss': MSELoss,
#     'batch_size': 128,
#     'hidden_layers': [400, 300],
#     'action_bounds': 1,
#     'max_grad_norm': 1.,
#     'activ_func': F.tanh,
#     'batch_normalization': True,
# }

# DQN_CONFIG = {
#     'replace_target_iter':600,
#     'e_greedy':0.9,
#     'e_greedy_increment':None,
#     'optimizer': optim.RMSprop,
#     'loss' : MSELoss,
#     'batch_size': 32,
# }
#
# DDPG_CONFIG = {
#     'steps_per_iter': 50,
#     'learn_start_step': 10000,
#     'batch_size': 128,
#     'reward_decay': 0.99,
#     'tau' : 0.005,
#     'noise_var' : 0.3,
#     'noise_min' : 0.01,
#     'noise_decrease' : 2e-5,
#     'optimizer': optim.Adam,
#     'v_optimizer': optim.Adam,
#     'lr': 1e-4,
#     'value_lr' : 1e-3,
#     'hidden_layers': [400, 300],
#     'hidden_layers_v' : [400, 300],
#     'value_loss_func': MSELoss,
#     'activ_func': F.relu,
#     'out_activ_func': F.tanh,
#     'action_bounds':1,
#     'max_grad_norm': None,
# }
#
# TD3_CONFIG = {
#     'actor_delayed_steps': 2,
#     'smooth_epsilon': 0.5,
#     'smooth_noise': 0.2,
# }
#
#
# PPO_CONFIG = {
#     'nbatch_per_iter': 32,
#     'updates_per_iter': 10,
#     'clip_epsilon': 0.2,
#     'lr': 3e-4,
#     'v_coef': 0.5,
# }
# PPO_CONFIG['value_lr'] = PPO_CONFIG['lr']
#
# AdaptiveKLPPO_CONFIG = {
#     'init_beta':3.,
#     'nbatch_per_iter': 32,
#     'updates_per_iter': 10,
#     'lr': 3e-4,
#     'v_coef': 0.5,
# }
# AdaptiveKLPPO_CONFIG['value_lr'] = AdaptiveKLPPO_CONFIG['lr']
#
#
