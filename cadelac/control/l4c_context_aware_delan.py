import torch
import torch.nn as nn

import numpy as np

import casadi as cs
import l4casadi as l4c

from l4casadi.naive.nn import activation as activations
from l4casadi.naive.nn.linear import Linear as l4c_Linear

from cadelac.learning.models.context_aware_delan import LSTMModel, TCNModel

class ComponentNNNaive(l4c.naive.NaiveL4CasADiModule):
    def __init__(self, n_input, n_ouput, net_arch = None, 
                 n_enc_input = 1, activation_name = 'Tanh'):
        super(ComponentNNNaive, self).__init__()

        self.net_arch = net_arch
        self.n_input = n_input
        self.n_ouput = n_ouput
        self.n_enc_input = n_enc_input
        self.activation_name = activation_name
        self.apply_tf = True

        self.n_output = n_ouput

        if self.activation_name is None:
            self.act = lambda x: x
        elif type(self.activation_name) is str:
            self.act = getattr(activations, self.activation_name)()
        else:
            self.act = self.activation_name

        ## Create Networks
        self.layers = []
        # Create Input Layer
        input_layer_size = n_input
        if self.apply_tf:
            input_layer_size = 2 * input_layer_size

        if self.n_enc_input > 1:
            input_layer_size += self.n_enc_input
        
        self.input_layer_size = input_layer_size

        prev_size = input_layer_size
        for hidden_size in self.net_arch:
            self.layers.append(l4c_Linear(prev_size, hidden_size))  # Linear layer
            self.layers.append(self.act)  # Activation function
            prev_size = hidden_size  # Update previous layer size
        # Create Output Layer
        self.layers.append(l4c_Linear(prev_size, self.n_output))

        self.net = nn.Sequential(*self.layers)

    def input_tf(self, input):
        return cs.vertcat(cs.cos(input), cs.sin(input))

    def forward(self, input):
        # input = input
        input_q = input[:self.n_input]
        input_enc = input[self.n_input:]

        if self.apply_tf:
            input_q = self.input_tf(input_q)
        input = input_q

        if self.n_enc_input > 1:
            input = cs.vertcat(input, input_enc)

        return self.net(input).reshape((1,-1))

class ComponentNN(nn.Module):
    """
    Equivalent implementation of ComponentNN from cadelac.learning.models.context_aware_delan.
    """
    def __init__(self, n_input, n_ouput, net_arch, 
                 n_enc_input = 1, activation_name = 'Tanh'):
        super(ComponentNN, self).__init__()

        self.net_arch = net_arch
        self.n_input = n_input
        self.n_ouput = n_ouput
        self.n_enc_input = n_enc_input
        self.activation_name = activation_name
        self.apply_tf = True

        self.n_output = n_ouput

        if self.activation_name == 'Tanh':
            self.act = nn.Tanh()
        elif self.activation_name == 'ReLu':
            self.act = nn.ReLU()
        else:
            raise AssertionError

        ## Create Networks
        self.layers = []
        # Create Input Layer
        input_layer_size = n_input
        if self.apply_tf:
            input_layer_size = 2 * input_layer_size

        if self.n_enc_input > 1:
            input_layer_size += self.n_enc_input

        self.input_layer_size = input_layer_size

        prev_size = input_layer_size
        for hidden_size in self.net_arch:
            self.layers.append(torch.nn.Linear(prev_size, hidden_size))
            self.layers.append(self.act)
            prev_size = hidden_size
        # Create Output Layer
        self.layers.append(torch.nn.Linear(prev_size, self.n_output))
        
        self.net = nn.Sequential(*self.layers)
    
    def input_tf(self, input):
        return torch.cat([torch.cos(input), torch.sin(input)], axis=-1)

    def forward(self, input):
        input = input.flatten()
        input_q = input[:self.n_input]
        input_enc = input[self.n_input:]

        if self.apply_tf:
            input_q = self.input_tf(input_q)
        input = input_q

        # Use encoding input
        if self.n_enc_input > 1:
            input = torch.cat((input, input_enc), dim=-1)

        # Hack to proper compute the jacobians from the aprox solution
        return self.net(input).reshape((1,-1))

class L4CContextAwareDeLaN():
    def __init__(self, torch_model, n_dof = 2, 
                 n_enc_input = 1, device = 'cpu'):
        
        self.n_dof = n_dof
        self.n_enc_input = n_enc_input

        hyper = torch_model['hyper']
        self.net_arch_inertia = hyper.get('net_arch_inertia', None)
        self.net_arch_pot = hyper.get('net_arch_pot', None)
        self.non_linearity = hyper['activation']
        self.gain_hidden = hyper['gain_hidden']
        self.gain_output = hyper['gain_output']
        self._epsilon = hyper['diagonal_epsilon']
        self.softplus_beta = 1.0
        self.device = device
        self.hist_length = hyper.get('hist_length', 0)
        self.history_encoder = hyper.get('history_encoder', 'lstm')  # Default to LSTM for backward compatibility

        self.state_dict = torch_model['state_dict']

        # Compute non-zero elements of L:
        self.l_output_size = int((self.n_dof ** 2 + self.n_dof) / 2)
        self.l_diag_size = self.n_dof
        self.l_lower_size = self.l_output_size - self.n_dof

        # Indices for matrix version of l
        self.idx_l_diag = np.diag_indices(self.n_dof)
        self.idx_l_off_diag = np.tril_indices(self.n_dof, -1)

        # Indices for vector version l
        # Calculate the indices of the diagonal elements of L:
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1

        # Calculate the indices of the off-diagonal elements of L:
        idx_tril = np.extract([x not in idx_diag for x in np.arange(self.l_output_size)], np.arange(self.l_output_size))

        # Indexing for concatenation of l_o  and l_d
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]

        ## Torch based networks - used for inference and test
        self.inertia_net = ComponentNN(self.n_dof, self.l_output_size, self.net_arch_inertia, self.n_enc_input)
        self.potential_net = ComponentNN(self.n_dof, 1, self.net_arch_pot, self.n_enc_input)

        inertia_dict = {}
        potential_dict = {}
        for key, values in self.state_dict.items():
            if 'inertia' in key:
                inertia_dict.update({key.removeprefix('inertia_net.'): values})
            if 'potential' in key:
                potential_dict.update({key.removeprefix('potential_net.'): values})

        # Filter only network's parameters
        cmp_inertia_dict = self.inertia_net.state_dict()
        for key1, value1 in inertia_dict.items():
            if key1 in cmp_inertia_dict:  # Ensure key1 exists in cmp_inertia_dict
                cmp_inertia_dict[key1] = value1

        cmp_potential_dict = self.potential_net.state_dict()
        for key1, value1 in potential_dict.items():
            if key1 in cmp_potential_dict:  # Ensure key1 exists in cmp_inertia_dict
                cmp_potential_dict[key1] = value1

        self.inertia_net.load_state_dict(cmp_inertia_dict)
        self.potential_net.load_state_dict(cmp_potential_dict)

        self.l4c_inertia_net = l4c.L4CasADi(self.inertia_net, name='inertia_net', model_expects_batch_dim=False)
        self.l4c_potential_net = l4c.L4CasADi(self.potential_net, name='potential_net', model_expects_batch_dim=False)


        ## L4C Naive networks - used for control and integrated with acados
        self.inertia_net_nn_l4c_naive_comp = ComponentNNNaive(self.n_dof, self.l_output_size, self.net_arch_inertia, self.n_enc_input)
        self.inertia_net_nn_l4c_naive_comp.load_state_dict(cmp_inertia_dict)
        self.inertia_net_naive_comp = l4c.L4CasADi(self.inertia_net_nn_l4c_naive_comp, name='inertia_net_nn_l4c_naive_comp', model_expects_batch_dim=False)

        self.potential_net_nn_l4c_naive_comp = ComponentNNNaive(self.n_dof, 1, self.net_arch_pot, self.n_enc_input)
        self.potential_net_nn_l4c_naive_comp.load_state_dict(cmp_potential_dict)
        self.potential_net_naive_comp = l4c.L4CasADi(self.potential_net_nn_l4c_naive_comp, name='potential_net_nn_l4c_naive_comp', model_expects_batch_dim=False)

        ## Casadi
        q_eval_delan = cs.MX.sym("q_eval_delan", self.n_dof, 1)
        x_eval_delan = q_eval_delan

        if self.n_enc_input > 1:
            enc_eval_delan = cs.MX.sym("enc", self.n_enc_input, 1)
            x_eval_delan = cs.vertcat(x_eval_delan, enc_eval_delan)

        # History Encoder (LSTM or TCN)
        if self.hist_length > 0:
            self.n_lstm_input = hyper['n_lstm_input']
            self.n_lstm_hidden = hyper['n_lstm_hidden']
            self.n_lstm_depth = hyper['n_lstm_depth']
            
            # Load encoder based on type
            if self.history_encoder == 'lstm':
                self.hist_encoder = LSTMModel(self.n_lstm_input, self.n_lstm_hidden, self.n_enc_input, self.n_lstm_depth)
            elif self.history_encoder == 'tcn':
                tcn_channels = hyper.get('tcn_channels', [32, 32, 16, 16])
                tcn_kernel_size = hyper.get('tcn_kernel_size', 3)
                tcn_dropout = hyper.get('tcn_dropout', 0.1)
                self.hist_encoder = TCNModel(self.n_lstm_input, self.n_enc_input, 
                                            num_channels=tcn_channels,
                                            kernel_size=tcn_kernel_size,
                                            dropout=tcn_dropout)
            elif self.history_encoder == 'none':
                self.hist_encoder = None
            else:
                raise ValueError(f"Unknown history_encoder: {self.history_encoder}")
            
            # Load encoder weights
            if self.hist_encoder is not None:
                encoder_dict = {}
                for key, values in self.state_dict.items():
                    if 'lstm' in key:
                        encoder_dict.update({key.removeprefix('lstm.'): values})
                self.hist_encoder.load_state_dict(encoder_dict)
            
            # Keep lstm attribute for backward compatibility
            self.lstm = self.hist_encoder

        if self.device == 'cuda':
            self.inertia_net = self.inertia_net.to('cuda')
            self.potential_net = self.potential_net.to('cuda')
            if hasattr(self, 'hist_encoder') and self.hist_encoder is not None:
                self.hist_encoder = self.hist_encoder.to('cuda')

    def lower_tri_inertia_fn(self, q, enc_input):
        output = self.inertia_net(q, enc_input).view(-1)
        l_diagonal, l_off_diagonal = torch.split(output, [self.l_diag_size, self.l_lower_size], dim=-1)

        # Ensure positive diagonal
        l_diagonal = torch.nn.Softplus(self.softplus_beta)(l_diagonal) + self._epsilon

        # Assemble l
        l_vec = torch.cat((l_diagonal, l_off_diagonal), -1)[..., self._idx]

        l = torch.zeros((self.n_dof, self.n_dof)).to(l_vec.device)
        tril_indices = torch.tril_indices(self.n_dof, self.n_dof)
        l = l.index_put((tril_indices[0], tril_indices[1]), l_vec)

        # Returning twice to use as aux variable when computing the jacobian
        return l, l

    def SoftPlusCS(self, x, beta = 1.0):
        return 1.0 / beta * cs.log(1 + cs.exp(beta * x))

    def lower_tri_inertia_fn_cs(self, nn_output):

        l_diagonal, l_off_diagonal = cs.vertsplit(nn_output.T, [0, self.l_diag_size, self.l_output_size])

        # Ensure positive diagonal
        l_diagonal = self.SoftPlusCS(l_diagonal) + self._epsilon

        # # Diagonal assignment
        l = cs.diag(l_diagonal)
        for index in range(self.l_lower_size):
            l[self.idx_l_off_diag[0][index], self.idx_l_off_diag[1][index]] = l_off_diagonal[index]

        return l.reshape((-1,1))

    def inertia_nn_out(self, q, enc_input):
        l_raw = self.inertia_net(q, enc_input)
        return l_raw, l_raw

    def potential_nn_out(self, q, enc_input):
        V = self.potential_net(q, enc_input)
        return V, V

    def get_jac_value_inertia_torch(self, q_np, enc_input_np):

        q = torch.from_numpy(q_np).float()
        if self.device == 'cuda':
            q = q.to('cuda')

        if enc_input_np is not None:
            enc_input = torch.from_numpy(enc_input_np).float()
            if self.device == 'cuda':
                enc_input = enc_input.to('cuda')
        else:
            enc_input = None

        if self.n_enc_input == 1:
            (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.lower_tri_inertia_fn, argnums=0, has_aux=True), in_dims=(0, None))(q, enc_input)
            (dVdq, V) = torch.func.vmap(torch.func.jacfwd(self.potential_nn_out, argnums=0, has_aux=True), in_dims=(0, None))(q, enc_input)

        else:
            (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.lower_tri_inertia_fn, argnums=0, has_aux=True))(q, enc_input)
            (dVdq, V) = torch.func.vmap(torch.func.jacfwd(self.potential_nn_out, argnums=0, has_aux=True))(q, enc_input)

        return l, dldq, V, dVdq


    def eval_hist_encoder_np(self, input_np):
        """Unified method to evaluate history encoder (LSTM or TCN) from numpy input."""
        if self.hist_length == 0:
            return np.zeros(self.n_enc_input)
        
        if self.hist_encoder is None:
            return np.zeros(self.n_enc_input)
        
        input = torch.from_numpy(input_np).float().view(1, self.hist_length, -1)
        if self.device == 'cuda':
            input = input.to('cuda')
        output = self.hist_encoder(input).view(-1)
        return output.cpu().detach().numpy()

    def eval_lstm_np(self, input_np):
        """Backward compatibility method - calls eval_hist_encoder_np."""
        return self.eval_hist_encoder_np(input_np)