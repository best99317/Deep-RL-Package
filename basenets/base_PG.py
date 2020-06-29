import torch
import numpy as np
from .MLP import MLP
from basenets.Conv import Conv
from torch import nn
import torch.nn.functional as F


class FCPG_Gaussian(MLP):
    def __init__(self,
                 num_input_feats,
                 num_output_feats,
                 sigma,
                 num_hiddens=None,
                 activ_func=torch.tanh,
                 batch_normalization=False,
                 out_active=None,
                 out_scaler=None,
                 initializer="orthogonal",
                 initializer_param=None
                 ):
        if initializer_param is None:
            initializer_param = {"gain": np.sqrt(2), "last_gain": 0.1}
        if num_hiddens is None:
            num_hiddens = [30]
        self.num_output_feats = num_output_feats
        super(FCPG_Gaussian, self).__init__(
            num_input_feats,  # input dim
            num_output_feats,  # output dim
            num_hiddens,  # hidden unit number list
            activ_func,
            batch_normalization,
            out_active,
            out_scaler,
            initializer,
            initializer_param=initializer_param,
        )
        self.log_std = nn.Parameter(torch.log(sigma * torch.ones(num_output_feats) + 1e-8))

    def forward(self, x, other_data=None):
        x = MLP.forward(self, x, other_data)
        return x, self.log_std.expand_as(x), torch.exp(self.log_std).expand_as(x)

    def cuda(self, device=None):
        self.log_std.cuda()
        return self._apply(lambda t: t.cuda(device))


class FCPG_Softmax(MLP):
    def __init__(self,
                 num_input_feats,    # input dim
                 num_output_feats,   # output dim
                 num_hiddens=None,  # hidden unit number list
                 activ_func=torch.tanh,
                 batch_normalization=False,
                 out_active=F.softmax,
                 out_scaler=None,
                 initializer="orthogonal",
                 initializer_param=None
                 ):
        if initializer_param is None:
            initializer_param = {"gain": np.sqrt(2), "last_gain": 0.1}
        if num_hiddens is None:
            num_hiddens = [10]
        self.num_output_feats = num_output_feats
        super(FCPG_Softmax, self).__init__(
                 num_input_feats,    # input dim
                 num_output_feats,   # output dim
                 num_hiddens,  # hidden unit number list
                 activ_func,
                 batch_normalization,
                 out_active,
                 out_scaler,
                 initializer,
                 initializer_param=initializer_param,
                 )


class ConvPG_Softmax(Conv):
    def __init__(self,
                 num_input_feats,    # input dim
                 num_output_feats,   # output dim
                 k_sizes=None,
                 channels=None,
                 strides=None,
                 fcs=None,  # hidden unit number list
                 activ_func=torch.relu,
                 batch_normalization=False,
                 out_active=torch.softmax,
                 out_scaler=None,
                 initializer="xavier",
                 initializer_param=None
                 ):
        if initializer_param is None:
            initializer_param = {}
        if fcs is None:
            fcs = [32, 32, 32]
        if strides is None:
            strides = [4, 2, 2]
        if channels is None:
            channels = [8, 16, 16]
        if k_sizes is None:
            k_sizes = [8, 4, 3]
        self.num_output_feats = num_output_feats
        super(ConvPG_Softmax, self).__init__(
                 num_input_feats,    # input dim
                 num_output_feats,   # output dim
                 k_sizes,
                 channels,
                 strides,
                 fcs,
                 activ_func,
                 batch_normalization,
                 out_active,
                 out_scaler,
                 initializer,
                 initializer_param=initializer_param,
                 )


# TODO: support multi-layer value function in which action is concat before the final layer
class FC_VALUE_FUNC(MLP):
    def __init__(self,
                 num_input_feats,
                 num_hiddens=None,
                 activ_func=torch.tanh,
                 batch_normalization=False,
                 out_active=None,
                 out_scaler=None,
                 initializer="orthogonal",
                 initializer_param=None
                 ):
        super(FC_VALUE_FUNC, self).__init__(
                 num_input_feats,
                 1,
                 num_hiddens,
                 activ_func,
                 batch_normalization,
                 out_active,
                 out_scaler,
                 initializer,
                 initializer_param=initializer_param,
                 )
        if initializer_param is None:
            initializer_param = {"gain": np.sqrt(2), "last_gain": 0.1}
        if num_hiddens is None:
            num_hiddens = [30]

