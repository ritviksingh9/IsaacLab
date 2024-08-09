from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.algos_torch.sac_helper import  SquashedNormal
from rl_games.common.layers.recurrent import  GRUWithDones, LSTMWithDones

def _create_initializer(func, **kwargs):
    return lambda v : func(v, **kwargs)   

class NetworkBuilder:
    def __init__(self, **kwargs):
        pass

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class BaseNetwork(nn.Module):
        def __init__(self, **kwargs):
            nn.Module.__init__(self, **kwargs)

            self.activations_factory = object_factory.ObjectFactory()
            self.activations_factory.register_builder('relu', lambda **kwargs : nn.ReLU(**kwargs))
            self.activations_factory.register_builder('tanh', lambda **kwargs : nn.Tanh(**kwargs))
            self.activations_factory.register_builder('sigmoid', lambda **kwargs : nn.Sigmoid(**kwargs))
            self.activations_factory.register_builder('elu', lambda  **kwargs : nn.ELU(**kwargs))
            self.activations_factory.register_builder('selu', lambda **kwargs : nn.SELU(**kwargs))
            self.activations_factory.register_builder('swish', lambda **kwargs : nn.SiLU(**kwargs))
            self.activations_factory.register_builder('gelu', lambda **kwargs: nn.GELU(**kwargs))
            self.activations_factory.register_builder('softplus', lambda **kwargs : nn.Softplus(**kwargs))
            self.activations_factory.register_builder('None', lambda **kwargs : nn.Identity())

            self.init_factory = object_factory.ObjectFactory()
            #self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
            self.init_factory.register_builder('const_initializer', lambda **kwargs : _create_initializer(nn.init.constant_,**kwargs))
            self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
            self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_normal_,**kwargs))
            self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_uniform_,**kwargs))
            self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs : _create_initializer(torch_ext.variance_scaling_initializer,**kwargs))
            self.init_factory.register_builder('random_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.uniform_,**kwargs))
            self.init_factory.register_builder('kaiming_normal', lambda **kwargs : _create_initializer(nn.init.kaiming_normal_,**kwargs))
            self.init_factory.register_builder('orthogonal', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
            self.init_factory.register_builder('default', lambda **kwargs : nn.Identity() )

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def _calc_input_size(self, input_shape,cnn_layers=None):
            if cnn_layers is None:
                assert(len(input_shape) == 1)
                return input_shape[0]
            else:
                return nn.Sequential(*cnn_layers)(torch.rand(1, *(input_shape))).flatten(1).data.size(1)

        def _noisy_dense(self, inputs, units):
            return layers.NoisyFactorizedLinear(inputs, units)

        def _build_rnn(self, name, input, units, layers):
            if name == 'identity':
                return torch_ext.IdentityRNN(input, units)
            if name == 'lstm':
                return LSTMWithDones(input_size=input, hidden_size=units, num_layers=layers)
            if name == 'gru':
                return GRUWithDones(input_size=input, hidden_size=units, num_layers=layers)

        def _build_sequential_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func,
        norm_only_first_layer=False, 
        norm_func_name = None):
            print('build mlp:', input_size)
            in_size = input_size
            layers = []
            need_norm = True
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))

                if not need_norm:
                    continue
                if norm_only_first_layer and norm_func_name is not None:
                   need_norm = False 
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit

            return nn.Sequential(*layers)

        def _build_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func, 
        norm_only_first_layer=False,
        norm_func_name = None,
        d2rl=False):
            if d2rl:
                act_layers = [self.activations_factory.create(activation) for i in range(len(units))]
                return D2RLNet(input_size, units, act_layers, norm_func_name)
            else:
                return self._build_sequential_mlp(input_size, units, activation, dense_func, norm_func_name = None,)

        def _build_conv(self, ctype, **kwargs):
            print('conv_name:', ctype)

            if ctype == 'conv2d':
                return self._build_cnn2d(**kwargs)
            if ctype == 'coord_conv2d':
                return self._build_cnn2d(conv_func=torch_ext.CoordConv2d, **kwargs)
            if ctype == 'conv1d':
                return self._build_cnn1d(**kwargs)

        def _build_cnn2d(self, input_shape, convs, activation, conv_func=torch.nn.Conv2d, norm_func_name=None):
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(conv_func(in_channels=in_channels, 
                out_channels=conv['filters'], 
                kernel_size=conv['kernel_size'], 
                stride=conv['strides'], padding=conv['padding']))
                conv_func=torch.nn.Conv2d
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch_ext.LayerNorm2d(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)

        def _build_cnn1d(self, input_shape, convs, activation, norm_func_name=None):
            print('conv1d input shape:', input_shape)
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(torch.nn.Conv1d(in_channels, conv['filters'], conv['kernel_size'], conv['strides'], conv['padding']))
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)


class A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):

            # Network structure:
            # obs is a dictionary, each gets processed according to the network in input_params.
            # Then features from each are flattened, concatenated, and go through an MLP.
            # for example
            #    camera -> CNN ->
            #                    |-> MLP -> action
            # joint_pos -> MLP ->

            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            # self.actor_cnn = nn.Sequential()
            # self.critic_cnn = nn.Sequential()

            self.input_networks_actor = nn.ModuleDict()
            self.input_networks_critic = nn.ModuleDict()

            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            # size of the preprocessing of inputs before the main MLP
            input_out_size = 0

            for input_name, input_config in self.input_params.items():
                has_cnn = 'cnn' in input_config
                has_mlp = 'mlp' in input_config


                member_input_shape = torch_ext.shape_whc_to_cwh(input_shape[input_name])
                networks_actor = []
                networks_critic = []

                # add CNN head if it is part of the config
                if has_cnn:
                    cnn_config = input_config['cnn']
                    cnn_args = {
                        'ctype' : cnn_config['type'],
                        'input_shape' : member_input_shape,
                        'convs' : cnn_config['convs'],
                        'activation' : cnn_config['activation'],
                        'norm_func_name' : self.normalization,
                    }
                    networks_actor.append(self._build_conv(**cnn_args))

                    if self.separate:
                        networks_critic.append(self._build_conv(**cnn_args))

                next_input_shape = self._calc_input_size(
                    member_input_shape, networks_actor[-1] if has_cnn else None
                )

                if has_mlp:
                    mlp_config = input_config['mlp']

                    mlp_args = {
                        'input_size': next_input_shape,
                        'units': mlp_config['units'],
                        'activation': mlp_config['activation'],
                        'norm_func_name': mlp_config.get('normalization', None),
                        'dense_func': torch.nn.Linear,
                        'd2rl': mlp_config.get('d2rl', False),
                        'norm_only_first_layer': self.norm_only_first_layer
                    }
                    networks_actor.append(self._build_mlp(**mlp_args))
                    if self.separate:
                        networks_critic.append(self._build_mlp(**mlp_args))
                    next_input_shape = mlp_config['units'][-1]

                # mlp?

                self.input_networks_actor[input_name] = nn.Sequential(*networks_actor)
                self.input_networks_critic[input_name] = nn.Sequential(*networks_critic)
                input_out_size += next_input_shape
            
            # in_mlp_shape = mlp_input_shape
            in_mlp_shape = input_out_size

            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                else:
                    rnn_in_size =  in_mlp_shape
                    in_mlp_shape = self.rnn_units

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)
                
                if self.is_aux:
                    mlp_args = {
                        'input_size': self.units[-1] + input_out_size,
                        'units': self.aux_units,
                        'activation': self.aux_activation,
                        'norm_func_name': self.aux_network.get('normalization', None),
                        'dense_func': torch.nn.Linear,
                        'd2rl': self.aux_is_d2rl,
                        'norm_only_first_layer': self.aux_norm_only_first_layer
                    }
                    
                    self.aux_mlp = self._build_mlp(**mlp_args)

                    self.aux_networks = nn.ModuleDict()

                    for output_name in self.aux_outputs:
                        assert len(input_shape[output_name]) == 1
                        aux_out_size = input_shape[output_name][0]
                        self.aux_networks[output_name] = nn.Sequential(
                            nn.Linear(self.aux_units[-1], aux_out_size),
                            self.activations_factory.create(self.aux_out_activation)
                        )
            else:
                if self.is_aux:
                    mlp_args = {
                        'input_size': self.rnn_units + input_out_size,
                        'units': self.aux_units,
                        'activation': self.aux_activation,
                        'norm_func_name': self.aux_network.get('normalization', None),
                        'dense_func': torch.nn.Linear,
                        'd2rl': self.aux_is_d2rl,
                        'norm_only_first_layer': self.aux_norm_only_first_layer
                    }
                    raise NotImplementedError('Currently aux loss only supported with RNN.')
            

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            # TODO - CNN init
            # if self.has_cnn:
            #     cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                #     cnn_init(m.weight)
                #     if getattr(m, "bias", None) is not None:
                #         torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

        def forward(self, obs_dict):
            

            obs = obs_dict['obs']           
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            # need to copy the dict here to prevent it from modifying the original dict when we permute
            obs = {**obs}
            # for input_name, input_config
            for input_name, input_config in self.input_params.items():
                if 'cnn' in input_config:
                    if len(obs[input_name].shape) == 4:
                        # for obs shape 4
                        # input expected shape (B, W, H, C)
                        # convert to (B, C, W, H)
                        obs[input_name] = obs[input_name].permute((0, 3, 1, 2))

            if self.separate:

                a_input_out_vals = []
                c_input_out_vals = []

                for input_name, input_config in self.input_params.items():
                    a_out_dim = self.input_networks_actor[input_name](obs[input_name])
                    c_out_dim = self.input_networks_critic[input_name](obs[input_name])
                    a_input_out_vals.append(a_out_dim.contiguous().view(a_out_dim.size(0), -1))
                    c_input_out_vals.append(c_out_dim.contiguous().view(c_out_dim.size(0), -1))

                a_out = torch.cat(a_input_out_vals, dim=-1)
                concatenated_input = a_out
                c_out = torch.cat(c_input_out_vals, dim=-1)

                if self.has_rnn:
                    if not self.is_rnn_before_mlp:
                        a_out_in = a_out
                        c_out_in = c_out
                        a_out = self.actor_mlp(a_out_in)
                        c_out = self.critic_mlp(c_out_in)

                        if self.rnn_concat_input:
                            a_out = torch.cat([a_out, a_out_in], dim=1)
                            c_out = torch.cat([c_out, c_out_in], dim=1)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0,1)

                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]                        
                    a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                    c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)
         
                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                    c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                    rnn_out = a_out

                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)
                        c_out = self.c_layer_norm(c_out)
                    
                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                    
                    
                    if self.is_aux:
                        self.last_aux_out = {}
                        aux_input = self.aux_mlp(
                            torch.cat(
                                [rnn_out, concatenated_input], dim=-1
                                
                            )
                        )
                        for output_name in self.aux_outputs:
                            self.last_aux_out[output_name] = self.aux_networks[output_name](aux_input)

                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)

                    ff_out = a_out
                    
                    if self.is_aux:
                        self.last_aux_out = {}
                        aux_input = self.aux_mlp(
                            torch.cat(
                                [ff_out, concatenated_input], dim=-1
                            )
                        )
                        for output_name in self.aux_outputs:
                            self.last_aux_out[output_name] = self.aux_networks[output_name](aux_input)
                            
                value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.fixed_sigma:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:

                input_out_vals = []

                for input_name, input_config in self.input_params.items():
                    input_out_vals.append(self.input_networks_actor[input_name](obs[input_name]).flatten(1))

                out = torch.cat(input_out_vals, dim=-1)
                concatenated_input = out

                if self.has_rnn:
                    out_in = out
                    if not self.is_rnn_before_mlp:
                        out_in = out
                        out = self.actor_mlp(out)
                        if self.rnn_concat_input:
                            out = torch.cat([out, out_in], dim=1)

                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)

                    if len(states) == 1:
                        states = states[0]

                    out = out.transpose(0,1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0, 1)
                    out, states = self.rnn(out, states, dones, bptt_len)
                    out = out.transpose(0,1)
                    out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)
                    rnn_out = out

                    if self.rnn_ln:
                        out = self.layer_norm(out)
                        
                    if self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                    if type(states) is not tuple:
                        states = (states,)
                    
                    if self.is_aux:
                        self.last_aux_out = {}
                        aux_input = self.aux_mlp(
                            torch.cat(
                                [rnn_out, concatenated_input], dim=-1
                                
                            )
                        )
                        for output_name in self.aux_outputs:
                            self.last_aux_out[output_name] = self.aux_networks[output_name](a_out)
                else:
                    out = self.actor_mlp(out)
                    ff_out = out
                    if self.is_aux:
                        self.last_aux_out = {}
                        aux_input = self.aux_mlp(
                            torch.cat(
                                [ff_out, concatenated_input], dim=-1
                                
                            )
                        )
                        for output_name in self.aux_outputs:
                            self.last_aux_out[output_name] = self.aux_networks[output_name](a_out)

                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, states
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.rnn_layers
            if self.rnn_name == 'identity':
                rnn_units = 1
            else:
                rnn_units = self.rnn_units
            if self.rnn_name == 'lstm':
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)                

        def load(self, params):
            self.separate = params.get('separate', False)

            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)

            # input params dictionary, where each can have MLP or CNN to process input
            self.input_params = params['inputs']
            
            self.is_aux = 'aux_outputs' in params
            if self.is_aux:
                self.aux_network = params['aux_network']
                self.aux_outputs = list(params['aux_outputs'].keys())

                self.aux_units = self.aux_network['mlp']['units']
                self.aux_activation = self.aux_network['mlp']['activation']
                self.aux_out_activation = self.aux_network['mlp']['out_activation']
                # self.aux_initializer = self.aux_network['mlp']['initializer']
                self.aux_is_d2rl = self.aux_network['mlp'].get('d2rl', False)
                self.aux_norm_only_first_layer = self.aux_network['mlp'].get('norm_only_first_layer', False)
            

            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete'in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False
     
            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.rnn_ln = params['rnn'].get('layer_norm', False)
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_concat_input = params['rnn'].get('concat_input', False)

            # if 'cnn' in params:
            #     self.has_cnn = True
            #     self.cnn = params['cnn']
            # else:
            #     self.has_cnn = False

    def build(self, name, **kwargs):
        net = A2CBuilder.Network(self.params, **kwargs)
        return net