import torch
import torch.nn as nn
import torch.optim as opt
from utils import cos


def get_mlp_layers(args):
    layers = []
    param_module_dict = {"mlp":args.model.mlp,"sparse_mlp":args.model.sparse_mlp}
    activation_fn_dict = {"relu":nn.ReLU, "cos":cos}
    output_activation_fn_dict = {"sigmoid":nn.Sigmoid,"cos":cos}
    param_module = param_module_dict[args.model.model_type]
    activation_fn = activation_fn_dict[param_module.activation]()
    output_activation_fn = output_activation_fn_dict[param_module.output_activation]()
    if(param_module.dropout == 0):
        if param_module.num_layer == 0:
            layers.append(nn.Linear(args.data.d, 1))
        else:
            layers.append(nn.Linear(args.data.d, param_module.hidden_dim))
            for _ in range(param_module.num_layer - 1):
                layers.append(activation_fn)
                layers.append(nn.Linear(param_module.hidden_dim, param_module.hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Linear(param_module.hidden_dim, 1))
    else:
        if param_module.num_layer == 0:
            layers.append(nn.Linear(args.data.d, 1))
        else:
            layers.append(nn.Linear(args.data.d, param_module.hidden_dim))
            for _ in range(param_module.num_layer - 1):
                layers.append(activation_fn)
                layers.append(nn.Dropout(param_module.dropout))
                layers.append(nn.Linear(param_module.hidden_dim, param_module.hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Linear(param_module.hidden_dim, 1))
    layers.append(output_activation_fn)
    return nn.Sequential(*layers)

def get_single_layer_mask(size, p):
    return torch.bernoulli(torch.ones(size)*p)


def get_mlp_mask(args):
    layers = []
    param_module_dict = {"sparse_mlp":args.model.sparse_mlp}
    param_module = param_module_dict[args.model.model_type]
    if param_module.num_layer > 0:
        layers.append(get_single_layer_mask((args.data.d, param_module.hidden_dim), param_module.sparsity))
        for _ in range(param_module.num_layer - 1):
            layers.append(get_single_layer_mask((param_module.hidden_dim, param_module.hidden_dim),  param_module.sparsity))
    return layers


class mlp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.network = get_mlp_layers(args)
    def forward(self, input):
        return self.network(input)
    
class sparse_mlp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.network = get_mlp_layers(args)
        self.mask = get_mlp_mask(args)
        for k, mask in enumerate(self.mask):
            self.mask[k] = mask.to(args.device)
    def forward(self, input):
        cnt = 0
        for n,p in self.network.named_parameters():
            if(cnt < len(self.mask)):
                if('weight' in n):
                    keys = n.split(".")
                    dut = self.network
                    for key in keys[:-1]:
                        dut = dut._modules[key]
                    dut.__dict__['_parameters'][keys[-1]] = self.mask[cnt].T*p
                    cnt += 1
        return self.network(input)
        
class cnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        param_module = args.model.cnn
        self.cnn = nn.Conv1d(1, param_module.out_channel, param_module.kernel_size, padding = (param_module.kernel_size - 1)//2)
        self.mlp = nn.Linear((args.data.d)*param_module.out_channel, 1)
        activation_fn_dict = {"relu":nn.ReLU, "cos":cos}
        output_activation_fn_dict = {"sigmoid":nn.Sigmoid,"cos":cos}
        self.activation_fn = activation_fn_dict[param_module.activation]()
        self.output_activation_fn = output_activation_fn_dict[param_module.output_activation]()

    def forward(self, input):
        input = input.unsqueeze(dim = 1)
        cnn_output = self.cnn(input)
        cnn_output = self.activation_fn(cnn_output)
        cnn_output = cnn_output.reshape((cnn_output.shape[0], -1))
        output = self.mlp(cnn_output)
        return self.output_activation_fn(output)