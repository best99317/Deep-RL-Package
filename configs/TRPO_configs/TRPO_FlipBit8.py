TRPOconfig = {
    'cg_damping': 1e-2,
    'GAE_lambda': 0.,
    'reward_decay': 0.9,
    'max_kl_divergence': 0.001,
    'hidden_layers': [64, 64],
    'hidden_layers_v': [64, 64],
    'max_grad_norm': None,
    'value_lr': 5e-4,
    'train_v_iters': 10,
    'lr_pi': 1e-3,
    'steps_per_iter': 128,
    'value_func_type': 'FC',
}
TRPOconfig['memory_size'] = TRPOconfig['steps_per_iter']
