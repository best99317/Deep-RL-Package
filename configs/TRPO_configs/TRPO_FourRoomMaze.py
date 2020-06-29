TRPOconfig = {
    'cg_damping': 1e-3,
    'GAE_lambda': 0.,
    'reward_decay': 0.95,
    'max_kl_divergence': 2e-5,
    'hidden_layers': [64, 64],
    'hidden_layers_v': [64, 64],
    'max_grad_norm': None,
    'value_lr': 5e-4,
    'train_v_iters': 10,
    'lr_pi': 1e-3,
    'steps_per_iter': 256,
    'value_func_type': 'FC',
}
TRPOconfig['memory_size'] = TRPOconfig['steps_per_iter']
