from torch import optim
from torch.nn import MSELoss

VPGconfig = {
    'max_grad_norm': 2,
    'steps_per_iter': 1024,
    'action_bounds': 1,
    'optimizer_pi': optim.Adam,
    'value_func_type': 'FC',
    'hidden_layers_v': [64, 64],
    'GAE_lambda': 0.95,            # HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION. 2016 ICLR
    'loss_func_v': MSELoss,
    'optimizer_v': optim.Adam,
    'lr_pi': 0.001,
    'value_lr': 0.001,
    # 'entropy_weight':0.0,
    'momentum_v': None,
    'momentum_pi': None,
    'init_noise': 1.,
}
VPGconfig['memory_size'] = VPGconfig['steps_per_iter']
