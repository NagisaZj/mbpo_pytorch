import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import gzip
import itertools
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from rlkit.torch.networks import RNN

device = torch.device('cuda')

num_train = 60000  # 60k train examples
num_test = 10000  # 10k test examples
train_inputs_file_path = './MNIST_data/train-images-idx3-ubyte.gz'
train_labels_file_path = './MNIST_data/train-labels-idx1-ubyte.gz'
test_inputs_file_path = './MNIST_data/t10k-images-idx3-ubyte.gz'
test_labels_file_path = './MNIST_data/t10k-labels-idx1-ubyte.gz'

BATCH_SIZE = 100


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        if hasattr(m.bias,'data'):
            m.bias.data.fill_(0.0)


class EnsembleRole(nn.Module):
    def __init__(self,ensemble_size=7, n_agents=8,role_dim=3,bias=False):
        super(EnsembleRole, self).__init__()
        self.n_agents = n_agents
        self.role_dim = role_dim
        self.ensemble_size = ensemble_size
        self.role_nets = nn.Parameter(torch.Tensor(ensemble_size, self.n_agents, self.role_dim))
        self.bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, self.role_dim))
        self.apply(init_weights)
        # self.role_nets=nn.ModuleList()
        # for i in range(self.ensemble_size):
        #     self.role_nets.append(nn.Linear(self.n_agents, self.role_dim).to(device))


    def forward(self):
        agent_ids = torch.eye(self.n_agents, device=device)
        # return agent_ids[None,:,:].repeat(self.ensemble_size,1,1)
        w_times_x = torch.matmul(torch.unsqueeze(agent_ids,0), self.role_nets)
        return torch.add(w_times_x, self.bias[:, None, :]) if self.bias else w_times_x
        # output_list = []
        # agent_ids = torch.eye(self.n_agents, device=device)
        # for i in range(self.ensemble_size):
        #     roles = self.role_nets[i](agent_ids)
        #     output_list.append(torch.unsqueeze(roles,0))
        #
        # return torch.cat(output_list,0) # [ensemble_size,n, role_size]

class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        self.has_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  if self.has_bias else w_times_x  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class SepEnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(SepEnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight2 = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
            self.bias2 = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input[:,:-1,:], self.weight)
        output=  torch.add(w_times_x, self.bias[:, None, :])  # w times x + b
        w_times_x2 = torch.bmm(input[:,-1:,:], self.weight2)
        output2 = torch.add(w_times_x2, self.bias2[:, None, :])
        return torch.cat([output,output2],dim=1)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ModularEnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(ModularEnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # x: [ensemble_size,batch_size,num_joints,obs_size]
        w_times_x = torch.matmul(input,self.weight.unsqueeze(1))
        outputs = torch.add(w_times_x,torch.unsqueeze(self.bias.unsqueeze(1),1))
        return outputs


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class SepModularEnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(SepModularEnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight2 = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
            self.bias2 = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # x: [ensemble_size,batch_size,num_joints,obs_size]
        w_times_x = torch.matmul(input[:,:,:-1,:],self.weight.unsqueeze(1))
        outputs = torch.add(w_times_x,torch.unsqueeze(self.bias.unsqueeze(1),1))
        w_times_x_2 = torch.matmul(input[:,:,-1:,:],self.weight2.unsqueeze(1))
        outputs2 = torch.add(w_times_x_2, torch.unsqueeze(self.bias2.unsqueeze(1), 1))
        # print(input.shape,w_times_x.shape,outputs.shape,w_times_x_2.shape,outputs2.shape)
        return torch.cat([outputs,outputs2],dim=2)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class EnsembleTransformer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleTransformer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.networks = nn.ModuleList()
        for i in range(self.ensemble_size):
            encoder_layers = TransformerEncoderLayer(self.in_features, 2, 32, 0.0)
            self.networks.append(
                TransformerEncoder(
                    encoder_layers,
                    3,
                    norm=nn.LayerNorm(self.in_features) if 1 else None,
                )
            )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = []
        # x: [ensemble_size,batch_size,num_joints,obs_size+1]
        for i in range(self.ensemble_size):
            this_input = torch.transpose(input[i,:,:,:],0,1)#[num_joints,batch_size,output_size]
            # print('input shape',this_input.shape)
            output = self.networks[i](this_input)#[num_joints,batch_size,output_size]
            output = output.permute(1,0,2)#[batch_size,num_joints,output_size]
            # print('output shape', output.shape)
            outputs.append(torch.unsqueeze(output,0))

        return torch.cat(outputs, dim=0)  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class EnsembleRNN(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleRNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.networks = nn.ModuleList()
        for i in range(self.ensemble_size):
            self.networks.append(
               RNN(hidden_sizes=[64, 64],
        input_size=in_features,
        output_size=out_features)
            )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = []
        # x: [ensemble_size,batch_size,num_joints,obs_size+1]
        for i in range(self.ensemble_size):
            this_input = input[i,:,:,:]
            # print('input shape',this_input.shape)
            output = self.networks[i].forward_seq(this_input)
            # print('output shape', output.shape)
            outputs.append(torch.unsqueeze(output,0))

        return torch.cat(outputs, dim=0)  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class EnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = state_size + reward_size
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()

class ModularEnsembleMessage(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False):
        super(ModularEnsembleMessage, self).__init__()
        self.num_joints = 8
        self.obs_dim = 14
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(self.obs_dim + action_size+hidden_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = state_size + reward_size
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.0001)


        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        message = torch.zeros([x.shape[0],x.shape[1],self.hidden_size]).to(device)
        for i in range(self.num_joints):
            input = torch.cat([x[:,:,i*self.obs_dim:(i+1)*self.obs_dim],message],dim=-1)
            nn1_output = self.swish(self.nn1(input))
            nn2_output = self.swish(self.nn2(nn1_output))
            nn3_output = self.swish(self.nn3(nn2_output))
            nn4_output = self.swish(self.nn4(nn3_output))
            nn5_output = self.nn5(nn4_output)
            message = nn5_output
        return message




    def train(self, loss):
        self.optimizer.zero_grad()


        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()


class ModularEnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3,
                 use_decay=False,single_state_size=None,num_joints=None):
        super(ModularEnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.single_state_size = state_size // num_joints
        self.state_size = state_size
        self.num_joints = num_joints
        self.ensemble_size = ensemble_size
        self.nn1 = ModularEnsembleFC( single_state_size+1, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleTransformer(hidden_size, hidden_size, ensemble_size, weight_decay=0.000025)
        # self.nn2 = ModularEnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.use_decay = use_decay

        self.output_dim = single_state_size + reward_size
        self.output_dim_real = state_size + reward_size
        # Add variance output
        self.nn5 = ModularEnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim_real)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim_real)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        state,action = x[...,:self.state_size],x[...,self.state_size:]
        # x: [ensemble_size,batch_size,obs_size*num_joints]
        # print(x.shape,state.shape,action.shape)
        state=state.reshape([state.shape[0],state.shape[1],self.num_joints,-1])
        action = action.reshape([action.shape[0],action.shape[1],self.num_joints,-1])
        x=torch.cat([state,action],dim=-1)
        # x: [ensemble_size,batch_size,num_joints,obs_size+1]
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn5_output = self.nn5(nn2_output)
        next_states, rewards = nn5_output[...,:self.single_state_size*2],nn5_output[...,self.single_state_size*2:(self.single_state_size*2+2)]
        next_states_mean,next_states_var = next_states[...,:self.single_state_size],next_states[...,self.single_state_size:]
        rewards_mean, rewards_var = rewards[..., :1], rewards[...,1:]
        next_states_mean = next_states_mean.reshape([next_states_mean.shape[0],next_states_mean.shape[1],-1])
        next_states_var = next_states_var.reshape([next_states_var.shape[0], next_states_var.shape[1], -1])
        rewards_mean =  rewards_mean[:,:,0,:]
        rewards_var = rewards_var[:, :, 0, :]
        # print(next_states.shape,rewards.shape,next_states_mean.shape,rewards_mean.shape)
        nn5_output = torch.cat([rewards_mean,next_states_mean,rewards_var,next_states_var],2)
        mean = nn5_output[:, :, :self.output_dim_real]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim_real:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        # print(mean.shape,logvar.shape,labels.shape)
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()

class RoleModularEnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-4,
                 use_decay=False,single_state_size=None,num_joints=None):
        super(RoleModularEnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.single_state_size = state_size // num_joints
        self.state_size = state_size
        self.num_joints = num_joints
        self.ensemble_size = ensemble_size
        role_dim = 8
        self.role_generator = EnsembleRole(ensemble_size=self.ensemble_size, n_agents=self.num_joints,role_dim=role_dim)
        self.encoder_net = EnsembleFC(role_dim, hidden_size * ( single_state_size+1), ensemble_size, weight_decay=0.000025,bias=False)
        # self.nn1 = ModularEnsembleFC( single_state_size+1, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleTransformer(hidden_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.use_decay = use_decay

        self.output_dim = single_state_size + reward_size
        self.output_dim_real = state_size + reward_size
        # Add variance output
        self.decoder_net = EnsembleFC(role_dim, hidden_size * (single_state_size + 1)*2, ensemble_size, weight_decay=0.000025,bias=False)
        # self.nn5 = ModularEnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim_real)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim_real)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # for p in self.parameters():
        #     print(p.shape)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        state,action = x[...,:self.state_size],x[...,self.state_size:]
        # x: [ensemble_size,batch_size,(obs_size+1)*num_joints]
        # print(x.shape,state.shape,action.shape)
        state=state.reshape([state.shape[0],state.shape[1],self.num_joints,-1])
        action = action.reshape([action.shape[0],action.shape[1],self.num_joints,-1])
        x=torch.cat([state,action],dim=-1)
        # x: [ensemble_size,batch_size,num_joints,obs_size+1]
        roles = self.role_generator.forward()#[ensemble_size,num_joints,role_size]
        encoder_params = self.encoder_net(roles).reshape(self.ensemble_size,self.num_joints,self.single_state_size+1,self.hidden_size) # [ensemble_size,num_joints,obs_size+1,hidden_size]
        x=x.unsqueeze(3)
        encoder_params = encoder_params.unsqueeze(1)# [ensemble_size,1,num_joints,obs_size+1,hidden_size]
        nn1_output = torch.matmul(x,encoder_params)# [ensemble_size,batch_size,num_joints,hidden_size]
        # print(x.shape,encoder_params.shape,nn1_output.shape)
        nn1_output = nn1_output.squeeze(3)
        # x: [ensemble_size,batch_size,num_joints,obs_size+1]
        nn1_output = self.swish(nn1_output)
        nn2_output = self.nn2(nn1_output)
        nn2_output = self.swish(nn2_output)

        decoder_params = self.decoder_net(roles).reshape(self.ensemble_size, self.num_joints, self.hidden_size,
                                                         -1)  # [ensemble_size,num_joints,hidden_size,output_size]
        nn2_output = nn2_output.unsqueeze(3)
        decoder_params = decoder_params.unsqueeze(1)  # [ensemble_size,1,num_joints,hidden_size,output_size]
        nn5_output = torch.matmul(nn2_output, decoder_params)  # [ensemble_size,batch_size,num_joints,output_size]
        # print(nn2_output.shape, decoder_params.shape, nn5_output.shape)
        nn5_output = nn5_output.squeeze(3)

        next_states, rewards = nn5_output[...,:self.single_state_size*2],nn5_output[...,self.single_state_size*2:(self.single_state_size*2+2)]
        next_states_mean,next_states_var = next_states[...,:self.single_state_size],next_states[...,self.single_state_size:]
        rewards_mean, rewards_var = rewards[..., :1], rewards[...,1:]
        next_states_mean = next_states_mean.reshape([next_states_mean.shape[0],next_states_mean.shape[1],-1])
        next_states_var = next_states_var.reshape([next_states_var.shape[0], next_states_var.shape[1], -1])
        rewards_mean =  rewards_mean[:,:,-1,:]
        rewards_var = rewards_var[:, :, -1, :]
        # print(next_states.shape,rewards.shape,next_states_mean.shape,rewards_mean.shape)
        nn5_output = torch.cat([rewards_mean,next_states_mean,rewards_var,next_states_var],2)
        mean = nn5_output[:, :, :self.output_dim_real]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim_real:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        # print(mean.shape,logvar.shape,labels.shape)
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()

class ModularEnsembleRNNModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3,
                 use_decay=False,single_state_size=None,num_joints=None):
        super(ModularEnsembleRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.single_state_size = state_size // num_joints
        self.state_size = state_size
        self.num_joints = num_joints
        self.ensemble_size = ensemble_size
        self.nn1 = SepModularEnsembleFC( single_state_size+1, hidden_size, ensemble_size, weight_decay=0.000025)
        # self.nn2 = EnsembleRNN(hidden_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleTransformer(hidden_size, hidden_size, ensemble_size, weight_decay=0.000025)
        # self.nn2 = ModularEnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.use_decay = use_decay

        self.output_dim = single_state_size + reward_size
        self.output_dim_real = state_size + reward_size
        # Add variance output
        self.nn5 = SepModularEnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim_real)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim_real)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        state,action = x[...,:self.state_size],x[...,self.state_size:]
        # x: [ensemble_size,batch_size,obs_size*num_joints]
        # print(x.shape,state.shape,action.shape)
        state=state.reshape([state.shape[0],state.shape[1],self.num_joints,-1])
        action = action.reshape([action.shape[0],action.shape[1],self.num_joints,-1])
        x=torch.cat([state,action],dim=-1)
        # x: [ensemble_size,batch_size,num_joints,obs_size+1]
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn5_output = self.nn5(nn2_output)
        # print(nn2_output.shape,nn5_output.shape)
        next_states, rewards = nn5_output[...,:self.single_state_size*2],nn5_output[...,self.single_state_size*2:(self.single_state_size*2+2)]
        next_states_mean,next_states_var = next_states[...,:self.single_state_size],next_states[...,self.single_state_size:]
        rewards_mean, rewards_var = rewards[..., :1], rewards[...,1:]
        next_states_mean = next_states_mean.reshape([next_states_mean.shape[0],next_states_mean.shape[1],-1])
        next_states_var = next_states_var.reshape([next_states_var.shape[0], next_states_var.shape[1], -1])
        rewards_mean =  rewards_mean[:,:,-1,:]
        rewards_var = rewards_var[:, :, -1, :]
        # print(next_states.shape,rewards.shape,next_states_mean.shape,rewards_mean.shape)
        nn5_output = torch.cat([rewards_mean,next_states_mean,rewards_var,next_states_var],2)
        mean = nn5_output[:, :, :self.output_dim_real]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim_real:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        # print(mean.shape,logvar.shape,labels.shape)
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()

class RoleModularEnsembleRNNModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-4,
                 use_decay=False,single_state_size=None,num_joints=None):
        super(RoleModularEnsembleRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.single_state_size = state_size // num_joints
        self.state_size = state_size
        self.num_joints = num_joints
        self.ensemble_size = ensemble_size
        role_dim = 4
        self.role_generator = EnsembleRole(ensemble_size=self.ensemble_size, n_agents=self.num_joints,role_dim=role_dim)
        self.encoder_net = EnsembleFC(role_dim, hidden_size * ( single_state_size+1), ensemble_size, weight_decay=0.000025)
        # self.nn1 = ModularEnsembleFC( single_state_size+1, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleRNN(hidden_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.use_decay = use_decay

        self.output_dim = single_state_size + reward_size
        self.output_dim_real = state_size + reward_size
        # Add variance output
        self.decoder_net = EnsembleFC(role_dim, hidden_size * (single_state_size + 1)*2, ensemble_size, weight_decay=0.000025)
        # self.nn5 = ModularEnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim_real)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim_real)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # for p in self.parameters():
        #     print(p.shape)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        state,action = x[...,:self.state_size],x[...,self.state_size:]
        # x: [ensemble_size,batch_size,(obs_size+1)*num_joints]
        # print(x.shape,state.shape,action.shape)
        state=state.reshape([state.shape[0],state.shape[1],self.num_joints,-1])
        action = action.reshape([action.shape[0],action.shape[1],self.num_joints,-1])
        x=torch.cat([state,action],dim=-1)
        # x: [ensemble_size,batch_size,num_joints,obs_size+1]
        roles = self.role_generator.forward()#[ensemble_size,num_joints,role_size]
        encoder_params = self.encoder_net(roles).reshape(self.ensemble_size,self.num_joints,self.single_state_size+1,self.hidden_size) # [ensemble_size,num_joints,obs_size+1,hidden_size]
        x=x.unsqueeze(3)
        encoder_params = encoder_params.unsqueeze(1)# [ensemble_size,1,num_joints,obs_size+1,hidden_size]
        nn1_output = torch.matmul(x,encoder_params)# [ensemble_size,batch_size,num_joints,hidden_size]
        # print(x.shape,encoder_params.shape,nn1_output.shape)
        nn1_output = nn1_output.squeeze(3)
        # x: [ensemble_size,batch_size,num_joints,obs_size+1]
        nn1_output = self.swish(nn1_output)
        nn2_output = self.nn2(nn1_output)
        nn2_output = self.swish(nn2_output)

        decoder_params = self.decoder_net(roles).reshape(self.ensemble_size, self.num_joints, self.hidden_size,
                                                         -1)  # [ensemble_size,num_joints,hidden_size,output_size]
        nn2_output = nn2_output.unsqueeze(3)
        decoder_params = decoder_params.unsqueeze(1)  # [ensemble_size,1,num_joints,hidden_size,output_size]
        nn5_output = torch.matmul(nn2_output, decoder_params)  # [ensemble_size,batch_size,num_joints,output_size]
        # print(nn2_output.shape, decoder_params.shape, nn5_output.shape)
        nn5_output = nn5_output.squeeze(3)

        next_states, rewards = nn5_output[...,:self.single_state_size*2],nn5_output[...,self.single_state_size*2:(self.single_state_size*2+2)]
        next_states_mean,next_states_var = next_states[...,:self.single_state_size],next_states[...,self.single_state_size:]
        rewards_mean, rewards_var = rewards[..., :1], rewards[...,1:]
        next_states_mean = next_states_mean.reshape([next_states_mean.shape[0],next_states_mean.shape[1],-1])
        next_states_var = next_states_var.reshape([next_states_var.shape[0], next_states_var.shape[1], -1])
        rewards_mean =  rewards_mean[:,:,-1,:]
        rewards_var = rewards_var[:, :, -1, :]
        # print(next_states.shape,rewards.shape,next_states_mean.shape,rewards_mean.shape)
        nn5_output = torch.cat([rewards_mean,next_states_mean,rewards_var,next_states_var],2)
        mean = nn5_output[:, :, :self.output_dim_real]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim_real:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        # print(mean.shape,logvar.shape,labels.shape)
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()

class EnsembleDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay)
        self.scaler = StandardScaler()

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5,epoch_step=None,cur_step=None,writer=None):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))
        writer.add_scalar('episode_model_counts', epoch,
                          cur_step)
        for i in range(len(holdout_mse_losses)):
            _, best = self._snapshots[i]
            writer.add_scalar('best mse loss%d'%i, best,
                            cur_step)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var


class ModularEnsembleDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False,num_joints = 8):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.num_joints = num_joints
        self.single_state_size = self.state_size // self.num_joints

        self.ensemble_model = ModularEnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay,single_state_size=self.single_state_size,num_joints=self.num_joints)
        self.scaler = StandardScaler()

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5,epoch_step=None,cur_step=None,writer=None):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))
        writer.add_scalar('episode_model_counts', epoch,
                          cur_step)
        for i in range(len(holdout_mse_losses)):
            _, best = self._snapshots[i]
            writer.add_scalar('best mse loss%d'%i, best,
                            cur_step)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var

class RoleModularEnsembleDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False,num_joints = 8):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.num_joints = num_joints
        self.single_state_size = self.state_size // self.num_joints

        self.ensemble_model = RoleModularEnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay,single_state_size=self.single_state_size,num_joints=self.num_joints)
        self.scaler = StandardScaler()

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5,epoch_step=None,cur_step=None,writer=None):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))
        writer.add_scalar('episode_model_counts', epoch,
                          cur_step)
        roles  = self.ensemble_model.role_generator.forward().cpu().data.numpy()
        np.save('/data2/zj/mbpo_pytorch/logs/modular-role-data/%d.pkl'%cur_step,roles)
        for i in range(len(holdout_mse_losses)):
            _, best = self._snapshots[i]
            writer.add_scalar('best mse loss%d'%i, best,
                            cur_step)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var

class ModularEnsembleRNNDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False,num_joints = 8):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.num_joints = num_joints
        self.single_state_size = self.state_size // self.num_joints

        self.ensemble_model = ModularEnsembleRNNModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay,single_state_size=self.single_state_size,num_joints=self.num_joints)
        self.scaler = StandardScaler()

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5,epoch_step=None,cur_step=None,writer=None):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))
        writer.add_scalar('episode_model_counts', epoch,
                          cur_step)
        for i in range(len(holdout_mse_losses)):
            _, best = self._snapshots[i]
            writer.add_scalar('best mse loss%d'%i, best,
                            cur_step)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var

class RoleModularEnsembleRNNDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False,num_joints = 8):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.num_joints = num_joints
        self.single_state_size = self.state_size // self.num_joints

        self.ensemble_model = RoleModularEnsembleRNNModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay,single_state_size=self.single_state_size,num_joints=self.num_joints)
        self.scaler = StandardScaler()

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5,epoch_step=None,cur_step=None,writer=None):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))
        writer.add_scalar('episode_model_counts', epoch,
                          cur_step)
        for i in range(len(holdout_mse_losses)):
            _, best = self._snapshots[i]
            writer.add_scalar('best mse loss%d'%i, best,
                            cur_step)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


def get_data(inputs_file_path, labels_file_path, num_examples):
    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_examples)
        data = np.frombuffer(buf, dtype=np.uint8) / 255.0
        inputs = data.reshape(num_examples, 784)

    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_examples)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int8)


def set_tf_weights(model, tf_weights):
    print(tf_weights.keys())
    pth_weights = {}
    pth_weights['max_logvar'] = tf_weights['BNN/max_log_var:0']
    pth_weights['min_logvar'] = tf_weights['BNN/min_log_var:0']
    pth_weights['nn1.weight'] = tf_weights['BNN/Layer0/FC_weights:0']
    pth_weights['nn1.bias'] = tf_weights['BNN/Layer0/FC_biases:0']
    pth_weights['nn2.weight'] = tf_weights['BNN/Layer1/FC_weights:0']
    pth_weights['nn2.bias'] = tf_weights['BNN/Layer1/FC_biases:0']
    pth_weights['nn3.weight'] = tf_weights['BNN/Layer2/FC_weights:0']
    pth_weights['nn3.bias'] = tf_weights['BNN/Layer2/FC_biases:0']
    pth_weights['nn4.weight'] = tf_weights['BNN/Layer3/FC_weights:0']
    pth_weights['nn4.bias'] = tf_weights['BNN/Layer3/FC_biases:0']
    pth_weights['nn5.weight'] = tf_weights['BNN/Layer4/FC_weights:0']
    pth_weights['nn5.bias'] = tf_weights['BNN/Layer4/FC_biases:0']
    for name, param in model.ensemble_model.named_parameters():
        if param.requires_grad:
            # print(name)
            print(param.data.shape, pth_weights[name].shape)
            param.data = torch.FloatTensor(pth_weights[name]).to(device).reshape(param.data.shape)
            pth_weights[name] = param.data
            print(name)


def main():
    torch.set_printoptions(precision=7)
    import pickle
    # Import MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    # train_inputs, train_labels = get_data(train_inputs_file_path, train_labels_file_path, num_train)
    # test_inputs, test_labels = get_data(test_inputs_file_path, test_labels_file_path, num_test)

    num_networks = 7
    num_elites = 5
    state_size = 17
    action_size = 6
    reward_size = 1
    pred_hidden_size = 200
    model = EnsembleDynamicsModel(num_networks, num_elites, state_size, action_size, reward_size, pred_hidden_size)

    # load tf weights and set it to be the inital weights for pytorch model
    with open('tf_weights.pkl', 'rb') as f:
        tf_weights = pickle.load(f)
    # set_tf_weights(model, tf_weights)
    # x = model.model_list[0].named_parameters()
    # for name, param in model.model_list[0].named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    # exit()
    BATCH_SIZE = 5250
    import time
    st_time = time.time()
    with open('test.npy', 'rb') as f:
        train_inputs = np.load(f)
        train_labels = np.load(f)
    for i in range(0, 1000, BATCH_SIZE):
        # train_inputs = np.random.random([BATCH_SIZE, state_size + action_size])
        # train_labels = np.random.random([BATCH_SIZE, state_size + 1])
        model.train(train_inputs, train_labels, holdout_ratio=0.2)
        # mean, var = model.predict(train_inputs[:100])
        # print(mean[0])
        # print(mean.mean().item())
        # print(var[0])
        # print(var.mean().item())
        # exit()
    print(time.time() - st_time)
    # for name, param in model.model_list[0].named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape,param)
    exit()
    # for i in range(0, 10000, BATCH_SIZE):
    #     model.train(Variable(torch.from_numpy(train_inputs[i:i + BATCH_SIZE])), Variable(torch.from_numpy(train_labels[i:i + BATCH_SIZE])))
    #
    # model.predict(Variable(torch.from_numpy(test_inputs[:1000])))


if __name__ == '__main__':
    main()
