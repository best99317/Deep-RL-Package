from torch.nn import functional as F

HTRPOconfig = {
    'cg_damping': 1e-3,
    'reward_decay': 0.98,
    'GAE_lambda': 0.,
    'max_kl_divergence': 2e-5,
    'per_decision': True,
    'weighted_is': True,
    'hidden_layers': [256, 256, 256],
    'hidden_layers_v': [256, 256, 256],
    'max_grad_norm': None,
    'value_lr': 5e-4,
    'train_v_iters': 20,
    # for comparison with HPG
    'lr_pi': 5e-4,
    # NEED TO FOCUS ON THESE PARAMETERS
    'using_hgf_goals': True,
    'steps_per_iter': 3200,
    'sampled_goal_num': 100,
    'value_func_type': 'FC',
    'out_act_func': F.tanh,
}
HTRPOconfig['memory_size'] = HTRPOconfig['steps_per_iter']
