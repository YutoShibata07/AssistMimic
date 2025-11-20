

import torch
import torch.nn as nn
from phc.learning.network_builder import NetworkBuilder
from collections import defaultdict
from rl_games.algos_torch import torch_ext
from tqdm import tqdm


class PNN(NetworkBuilder.BaseNetwork):

    def __init__(self, mlp_args, output_size=69, numCols=4, has_lateral=True, aux_mlp_units=None):
        super(PNN, self).__init__()
        self.numCols = numCols
        units = mlp_args['units']
        dense_func = mlp_args['dense_func']
        self.has_lateral = has_lateral

        self.actors = nn.ModuleList()
        for i in range(numCols):
            mlp = self._build_sequential_mlp(output_size, **mlp_args)
            self.actors.append(mlp)

        # Initialize zero_fc layers for fusion with aux_mlp intermediates
        if aux_mlp_units:
            self.zero_fc = nn.ModuleList()
            for layer_idx, aux_unit in enumerate(aux_mlp_units):
                # Match the corresponding PNN layer size
                if layer_idx < len(units):
                    pnn_unit = units[layer_idx]
                    # Create zero-initialized fusion layer
                    fusion_layer = nn.Linear(aux_unit, pnn_unit, bias=False)
                    nn.init.zeros_(fusion_layer.weight)
                    self.zero_fc.append(fusion_layer)
                else:
                    # If aux_mlp has more layers than PNN, skip
                    break
            print(f"Initialized {len(self.zero_fc)} zero_fc fusion layers")
        else:
            self.zero_fc = None

        if self.has_lateral:

            self.u = nn.ModuleList()

            for i in range(numCols - 1):
                self.u.append(nn.ModuleList())
                for j in range(i + 1):
                    u = nn.Sequential()
                    in_size = units[0]
                    for unit in units[1:]:
                        u.append(dense_func(in_size, unit, bias=False))
                        in_size = unit
                    u.append(dense_func(units[-1], output_size, bias=False))
                    #                     torch.nn.init.zeros_(u[-1].weight)
                    self.u[i].append(u)

    def freeze_pnn(self, idx):
        for param in self.actors[:idx].parameters():
            param.requires_grad = False
        if self.has_lateral:
            for param in self.u[:idx - 1].parameters():
                param.requires_grad = False

    def load_base_net(self, model_path, actors=1):
        # Set device and clear cache to prevent GPU0 memory imbalance
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(current_device)
            torch.cuda.empty_cache()
        # Load checkpoint with proper device mapping to avoid CUDA device mismatch
        if torch.cuda.is_available():
            current_device = f'cuda:{torch.cuda.current_device()}'
            checkpoint = torch.load(model_path, map_location=current_device)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        # Move checkpoint to CPU to prevent GPU0 memory bias
        if 'model' in checkpoint:
            cpu_model_state = {}
            for key, tensor in checkpoint['model'].items():
                if hasattr(tensor, 'cpu'):
                    cpu_model_state[key] = tensor.cpu()
                else:
                    cpu_model_state[key] = tensor
            checkpoint['model'] = cpu_model_state
        for idx in range(actors):
            self.load_actor(checkpoint, idx)

    def load_actor(self, checkpoint, idx=0):
        state_dict = self.actors[idx].state_dict()
        checkpoint_model = checkpoint['model']
        
        # Debug: Print available PNN keys
        pnn_keys = [k for k in checkpoint_model.keys() if 'pnn' in k and f'actors.{idx}' in k]
        print(f"DEBUG: Available PNN actor {idx} keys in checkpoint: {pnn_keys[:5]}...")  # Show first 5
        
        # Try different key prefixes for PNN structure
        possible_prefixes = [
            f'a2c_network.module.pnn.actors.{idx}.',
            f'a2c_network.pnn.actors.{idx}.',
            f'module.a2c_network.pnn.actors.{idx}.',
            f'pnn.actors.{idx}.'
        ]
        
        loaded = False
        for prefix in possible_prefixes:
            test_key = f'{prefix}0.weight'
            if test_key in checkpoint_model:
                # Load from PNN checkpoint structure with this prefix
                loaded_count = 0
                for key in state_dict.keys():
                    checkpoint_key = f'{prefix}{key}'
                    if checkpoint_key in checkpoint_model:
                        state_dict[key].copy_(checkpoint_model[checkpoint_key])
                        loaded_count += 1
                    else:
                        print(f"Warning: {checkpoint_key} not found in checkpoint")
                print(f"Successfully loaded {loaded_count}/{len(state_dict)} parameters for PNN actor {idx} using prefix '{prefix}'")
                loaded = True
                break
        
        if not loaded:
            # Fallback to old actor_mlp structure with different prefixes
            actor_mlp_prefixes = [
                'a2c_network.module.',
                'a2c_network.',
                'module.a2c_network.',
                ''
            ]
            
            for prefix in actor_mlp_prefixes:
                try:
                    state_dict['0.weight'].copy_(checkpoint_model[f'{prefix}actor_mlp.0.weight'])
                    state_dict['0.bias'].copy_(checkpoint_model[f'{prefix}actor_mlp.0.bias'])
                    state_dict['2.weight'].copy_(checkpoint_model[f'{prefix}actor_mlp.2.weight'])
                    state_dict['2.bias'].copy_(checkpoint_model[f'{prefix}actor_mlp.2.bias'])
                    state_dict['4.weight'].copy_(checkpoint_model[f'{prefix}mu.weight'])
                    state_dict['4.bias'].copy_(checkpoint_model[f'{prefix}mu.bias'])
                    print(f"Successfully loaded actor {idx} using fallback method with prefix '{prefix}'")
                    loaded = True
                    break
                except KeyError:
                    continue
            
            if not loaded:
                print(f"Error: Failed to load actor {idx} - no matching keys found")
                print(f"Available keys sample: {list(checkpoint_model.keys())[:10]}")
                print(f"Required keys not found in checkpoint")

    def _build_sequential_mlp(self, actions_num, input_size, units, activation, dense_func, norm_only_first_layer=False, norm_func_name=None, need_norm = True):
        print('build mlp:', input_size)
        in_size = input_size
        layers = []
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
            

        layers.append(nn.Linear(units[-1], actions_num))
        return nn.Sequential(*layers)

    def forward(self, x, idx=-1, aux_intermediates=None):
        if self.has_lateral:
            # idx == -1: forward all, output all
            # idx == others, forward till idx.
            if idx == 0:
                actions = self._forward_actor_with_aux(self.actors[0], x, aux_intermediates)
                return actions, [actions]
            else:
                if idx == -1:
                    idx = self.numCols - 1
                activation_cache = defaultdict(list)

                for curr_idx in range(0, idx + 1):
                    curr_actor = self.actors[curr_idx]
                    assert len(curr_actor) == 5  # Only support three MLPs right now

                    # Forward pass with aux_mlp fusion
                    activation_1 = self._forward_layer_with_aux(curr_actor[:2], x, aux_intermediates, 0)

                    acc_acts_1 = [self.u[curr_idx - 1][col_idx][0](activation_cache[0][col_idx]) for col_idx in range(len(activation_cache[0]))]  # curr_idx - 1 as we need to go to the previous coloumn's index to activate the weight

                    # Second layer with aux fusion
                    layer_2_input = curr_actor[2](activation_1) + sum(acc_acts_1)
                    activation_2 = self._forward_layer_with_aux(curr_actor[3:4], layer_2_input, aux_intermediates, 1)

                    # acc_acts_2 = [self.u[curr_idx - 1][col_idx][1](activation_cache[1][col_idx]) for col_idx in range(len(activation_cache[1]))]
                    # actions = curr_actor[4](activation_2) + sum(acc_acts_2)

                    actions = curr_actor[4](activation_2)  # disable action space transfer.

                    activation_cache[0].append(activation_1)
                    activation_cache[1].append(activation_2)
                    activation_cache[2].append(actions)

                return actions, activation_cache[2]
        else:
            if idx != -1:
                actions = self._forward_actor_with_aux(self.actors[idx], x, aux_intermediates)
                return actions, [actions]
            else:
                actions = [self._forward_actor_with_aux(self.actors[idx], x, aux_intermediates) for idx in range(self.numCols)]
                return actions, actions

    def _forward_actor_with_aux(self, actor, x, aux_intermediates=None):
        """Forward pass through actor with aux_mlp intermediate fusion"""
        if aux_intermediates is None or self.zero_fc is None:
            error_msg = "aux_intermediates is None or self.zero_fc is None"
            raise ValueError(error_msg)

        # Manual forward pass through each layer with aux fusion
        activation = x
        layer_idx = 0

        # Process layers in pairs (Linear + Activation + Optional Norm)
        i = 0
        while i < len(actor):
            # Linear layer
            if isinstance(actor[i], nn.Linear):
                activation = actor[i](activation)

                # # Add aux_mlp intermediate if available
                # if layer_idx < len(self.zero_fc) and layer_idx < len(aux_intermediates) and layer_idx > 3:
                #     aux_contribution = self.zero_fc[layer_idx](aux_intermediates[layer_idx])
                #     activation = activation + aux_contribution

                layer_idx += 1
            else:
                # Activation or normalization layer
                activation = actor[i](activation)

            i += 1

        return activation

    def _forward_layer_with_aux(self, layers, x, aux_intermediates=None, layer_idx=0):
        """Forward pass through a specific layer with aux fusion"""
        activation = x
        for layer in layers:
            if isinstance(layer, nn.Linear):
                activation = layer(activation)

                # Add aux_mlp intermediate if available
                # if (aux_intermediates is not None and
                #     self.zero_fc is not None and
                #     layer_idx < len(self.zero_fc) and
                #     layer_idx < len(aux_intermediates) and layer_idx > 3):
                #     aux_contribution = self.zero_fc[layer_idx](aux_intermediates[layer_idx])
                #     activation = activation + aux_contribution
            else:
                activation = layer(activation)

        return activation
