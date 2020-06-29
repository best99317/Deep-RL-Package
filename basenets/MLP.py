import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class MLP(nn.Module):
    def __init__(self,
                 num_input_feats,  # input dimension
                 num_output_feats,  # output dimension
                 num_hiddens=None,  # hidden unit number list
                 activ_func=torch.tanh,
                 batch_normalization=False,
                 out_active=None,  # activation applied to the output of the last layer
                 out_scaler=None,  # TODO: What is out_scaler for?
                 initializer="normal",  # Ways to initialize networks
                 initializer_param=None,  # only for orthogonal initializer
                 ):
        super(MLP, self).__init__()
        if num_hiddens is None:
            num_hiddens = [30]
        if initializer_param is None:
            initializer_param = {}
        self.activ_func = activ_func

        self.out_active = out_active
        if out_scaler is not None:
            if isinstance(out_scaler, (int, float)):  # check if out_scaler is int or float)
                self.out_scaler = Variable(torch.Tensor([out_scaler]))
            else:
                self.out_scaler = Variable(torch.Tensor(out_scaler))
        else:
            self.out_scaler = None
        input_list = [num_input_feats, ] + num_hiddens  # list of the input for each layer
        output_list = num_hiddens + [num_output_feats, ]  # list of the output for each layer
        self.layers = nn.ModuleList()  # create the network module
        for i, (input_units, output_units) in enumerate(zip(input_list, output_list)):
            if batch_normalization:
                bn_layer = nn.BatchNorm1d(input_units)
                bn_layer.weight.data.fill_(1)
                bn_layer.bias.data.fill_(0)
                self.layers.append(bn_layer)
            linear_layer = nn.Linear(input_units, output_units)

            # Different ways to initialize the network: Normal, Uniform, Orthogonal, Xavier, Kaiming
            nn.init.constant_(linear_layer.bias, 0)
            last_layer = (i == len(output_list) - 1)
            if initializer == "normal":
                if not last_layer:
                    var = initializer_param['var'] if 'var' in initializer_param.keys() else 1. / np.sqrt(input_units)
                    nn.init.normal_(linear_layer.weight, 0, var)
                else:
                    var = initializer_param['last_var'] if 'last_var' in initializer_param.keys() \
                        else 1. / np.sqrt(input_units)
                    nn.init.normal_(linear_layer.weight, 0, var)
                print("initializing layer " + str(i + 1) + " Method: Normal. Var: " + str(var))
            elif initializer == "uniform":
                if not last_layer:
                    lower = initializer_param['lower'] if 'lower' in initializer_param.keys() \
                        else -1. / np.sqrt(input_units)
                    upper = initializer_param['upper'] if 'upper' in initializer_param.keys() \
                        else 1. / np.sqrt(input_units)
                    nn.init.uniform_(linear_layer.weight, lower, upper)
                else:
                    lower = initializer_param['last_lower'] if 'last_lower' in initializer_param.keys() else -0.01
                    upper = initializer_param['last_upper'] if 'last_upper' in initializer_param.keys() else 0.01
                    nn.init.uniform_(linear_layer.weight, lower, upper)
                print("initializing layer " + str(i + 1)
                      + " Method: Uniform. Lower: " + str(lower) + ". Upper: " + str(upper))
            elif initializer == "orthogonal":
                if not last_layer:
                    gain = initializer_param['gain'] if 'gain' in initializer_param.keys() else np.sqrt(2)
                    nn.init.orthogonal_(linear_layer.weight, gain)
                    print("initializing layer " + str(i + 1) + " Method: Orthogonal. Gain: " + str(gain))
                else:
                    gain = initializer_param['last_gain'] if 'last_gain' in initializer_param.keys() else 0.1
                    nn.init.orthogonal_(linear_layer.weight, gain)
                print("initializing layer " + str(i + 1) + " Method: Orthogonal. Gain: " + str(gain))
            elif initializer == "xavier":
                gain = initializer_param['gain'] if 'gain' in initializer_param.keys() else 1
                nn.init.xavier_normal_(linear_layer.weight, gain)
                print("initializing layer " + str(i + 1) + " Method: Xavier. Gain: " + str(gain))
            elif initializer == "kaiming":
                a = initializer_param['a'] if 'a' in initializer_param.keys() else 0
                nn.init.kaiming_normal_(linear_layer.weight, a)
                print("initializing layer " + str(i + 1) + " Method: Kaiming_normal. a: " + str(a))
            else:
                assert 0, "please specify one initializer."
            self.layers.append(linear_layer)

    def forward(self, x, other_data=None):
        assert other_data is None or isinstance(other_data, torch.Tensor)  # TODO: What is other data for
        if other_data is not None:
            x = torch.cat((x, other_data), dim=-1)
        input_dim = x.dim()
        if input_dim == 1:
            x = x.unsqueeze(0)
        for num_layers, layer in enumerate(self.layers):
            x = layer(x)
            # the last layer
            if num_layers == len(self.layers) - 1:
                if self.out_active is not None:
                    if self.out_scaler is not None:
                        x = self.out_scaler.type_as(x) * self.out_active(x)
                    else:
                        x = self.out_active(x)
            else:
                x = self.activ_func(x)
        if input_dim == 1:
            x = x.squeeze(0)
        return x
