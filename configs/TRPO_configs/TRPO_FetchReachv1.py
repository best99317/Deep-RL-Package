TRPOconfig = {
    'cg_damping': 1e-3,
    'GAE_lambda': 0.,
    'reward_decay': 0.98,
    'max_kl_divergence': 2e-5,
    'hidden_layers': [256, 256, 256],
    'hidden_layers_v': [256, 256, 256],
    'max_grad_norm': None,
    'value_lr': 5e-4,
    'train_v_iters': 20,
    'lr_pi': 5e-4,
    'steps_per_iter': 3200,
    'value_func_type': 'FC',
}
TRPOconfig['memory_size'] = TRPOconfig['steps_per_iter']