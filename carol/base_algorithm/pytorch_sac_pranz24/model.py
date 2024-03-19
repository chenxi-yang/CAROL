import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import carol.modules.vrlnn as vrlnn

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, vrlnn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        # print(f"Q: {num_inputs}, {num_actions}, {hidden_dim}")
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        # with torch.no_grad():
        # import time
        # start_time = time.time()
        # print(f"In Q, state: {state.shape}; action: {action.shape}")
        # print(f"---Q forward---")
        # print(f"state: {state.detach().cpu().numpy().tolist()[0][0]}")
        # print(f"action: {action.detach().cpu().numpy().tolist()[0][0]}")
        xu = torch.cat([state, action], 1)
        # print(f"xu: {xu.detach().cpu().numpy().tolist()[0][0]}")
        # x1_before_relu = self.linear1(xu, test=True)
        # print(f"x1_before_relu: {x1_before_relu.detach().cpu().numpy().tolist()[0][0]}")
        # exit(0)
        # x1 = F.relu(x1_before_relu)
        # x1 = self.relu(self.linear1(xu))
        x1 = F.relu(self.linear1(xu))
        # print(f"x1: {x1.detach().cpu().numpy().tolist()[0][0]}")
        x1 = F.relu(self.linear2(x1))
        # print(f"Before linear3 {time.time() - start_time}")
        x1 = self.linear3(x1)
        # print(f"After linear3 {time.time() - start_time}")

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        # print(f"Before linear6 {time.time() - start_time}")
        x2 = self.linear6(x2)
        # print(f"---End Q forward---")
        # print(f"After linear6 {time.time() - start_time}")

        return x1, x2


class SymbolicQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(SymbolicQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = vrlnn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = vrlnn.Linear(hidden_dim, hidden_dim)
        self.linear3 = vrlnn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = vrlnn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = vrlnn.Linear(hidden_dim, hidden_dim)
        self.linear6 = vrlnn.Linear(hidden_dim, 1)

        self.relu = vrlnn.ReLU()
        self.apply(weights_init_)

    def forward(self, state, action):
        # print(f"---Symbolic Q forward---")
        # print(f"state: {state.detach().cpu().numpy().tolist()[0][0]}")
        # print(f"action: {action.detach().cpu().numpy().tolist()[0][0]}")
        if isinstance(state, torch.Tensor):
            xu = torch.cat([state, action], 1)
        else:
            xu = state.concatenate(action)
        # print(f"xu: {xu.detach().cpu().numpy().tolist()[0][0]}")
        # x1_before_relu = self.linear1(xu, test=False)
        # print(f"x1_before_relu: {x1_before_relu.detach().cpu().numpy().tolist()[0][0]}")
        # exit(0)
        # x1 = self.relu(x1_before_relu)
        x1 = self.relu(self.linear1(xu))
        # print(f"x1: {x1.detach().cpu().numpy().tolist()[0][0]}")
        x1 = self.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = self.relu(self.linear4(xu))
        x2 = self.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        # print(f"---End Symbolic Q forward---")

        return x1, x2
        # return the q1, q2's inf 
        # return x1.inf(), x2.inf()


class SymbolicGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        '''
        The symbolic gaussian policy where the mean and logstd are separate.
        Logstd is represented by an additional parameter.
        '''
        super(SymbolicGaussianPolicy, self).__init__()

        self.linear1 = vrlnn.Linear(num_inputs, hidden_dim)
        self.linear2 = vrlnn.Linear(hidden_dim, hidden_dim)
        self.relu = vrlnn.ReLU()

        self.mean_linear = vrlnn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        stdev_init = torch.zeros(num_actions)
        self.log_stdev = torch.nn.Parameter(stdev_init)
        # self.log_stdev_test = vrlnn.Linear(hidden_dim, num_actions)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            high = np.array(action_space.high)
            low = np.array(action_space.low)
            self.action_scale = torch.FloatTensor((high - low) / 2.0)
            self.action_bias = torch.FloatTensor((high + low) / 2.0)

    def forward(self, state):
        if isinstance(state, torch.Tensor):
            debug=False
        else: debug = False
        if debug:
            print(f"-------- in SymGaussian Forward ----------")
            print(f"shape: {state.shape}")
            print(f"state delta: {state.delta.min().item()}, {state.delta.max().item()}")
            print(f"state c: {state.c.min().item()}, {state.c.max().item()}")
            # print(f"state[0]: c:{state.c.detach().cpu().numpy().tolist()}, delta:{state.delta.detach().cpu().numpy().tolist()}")
        
        x = self.relu(self.linear1(state))

        if debug:
            print(f"x delta: {x.delta.min().item()}, {x.delta.max().item()}")
            print(f"x shape: {x.shape}")
            # print(f"x[0]: c:{x.c[0].detach().cpu().numpy().tolist()}, delta:{x.delta[0].detach().cpu().numpy().tolist()}")
            idx = torch.argmax(x.delta)
            print(f"x delta argmax: {torch.argmax(x.delta)}")
            # print(f"c: {x.c[int(idx/64)][idx%64]}, delta: {x.delta[int(idx/64)][idx%64]}")
            # print(f"mapped state: {state.c[int(idx/64)]}")
        # state: [-0.1119, -0.7625,  0.1640,  1.2762]
        # max delta: 0.16243940591812134
        x = self.relu(self.linear2(x))
        if debug:
            print(f"x delta: {x.delta.min().item()}, {x.delta.max().item()}")
        mean = self.mean_linear(x)
        if debug:
            # print(f"mapped mean: c: {mean.c[int(idx/256)]}; delta: {mean.delta[int(idx/256)]}")
            print(f"mean delta: {mean.delta.min().item()}, {mean.delta.max().item()}")
            # exit(0)
            mean_max_idx = torch.argmax(mean.delta)
            print(mean_max_idx, mean.shape)
            the_idx = int(mean_max_idx)
            print(f"mapped mean: c: {mean.c[the_idx]}; delta: {mean.delta[the_idx]}")
            print(f"the mapped state: {state.c[the_idx]} delta: {state.delta[the_idx]}")
        log_std = self.log_stdev
        #? test for the input dependent logstd
        # log_std_test = self.log_stdev_test(x)
        # if isinstance(log_std_test, torch.Tensor):
        #     log_std = log_std_test
        # else:
        #     log_std = log_std_test.c

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, mode=None, at=None):
        if mode == "RADIAL":
            # state is a box
            mean, log_std = self.forward(state)
            std = log_std.exp()
            # print(f"mean max delta: {mean.delta.max().item()}")
            # print(f"mean min delta: {mean.delta.min().item()}")
            # print(f"std: {std.item()}")
            d = (mean.sub_l(at)).mul(mean.sub_l(at)).mul(1/std.pow(2))
            d_max = d.c + d.delta
            d_min = d.inf()
            # print(f"d max delta: {d.delta.max().item()}")
            # print(f"d min delta: {d.delta.min().item()}")
            # print(f"count nan, d_max: {torch.isnan(d_max).sum()}, d_min: {torch.isnan(d_min).sum()}")
            # print(f"d_max: {d_max.detach().cpu().numpy().tolist()}")
            # print(f"d_min: {d_min.detach().cpu().numpy().tolist()}")
            # exit(0)

            log_p_min =  - (0.5 * d_max.sum(-1) \
                   + 0.5 * np.log(2.0 * np.pi) * at.shape[-1] \
                   + log_std.sum(-1))
            log_p_max =  - (0.5 * d_min.sum(-1) \
                   + 0.5 * np.log(2.0 * np.pi) * at.shape[-1] \
                   + log_std.sum(-1))
            # print(f"count nan, logpmax: {torch.isnan(log_p_max).sum()}, logpmin: {torch.isnan(log_p_min).sum()}")       
            # print(f"count nan, at: {torch.isnan(at).sum()}")  
            at_enforced_bound = at.tanh()
            at = at_enforced_bound
            # enforce action bound
            # tmp = self.action_scale * (1 - at.pow(2)) + epsilon
            # print(f"count nan, tmp: {torch.isnan(tmp).sum()}")
            # log_tmp = torch.log(tmp)
            # print(f"count nan, log_tmp: {torch.isnan(log_tmp).sum()}")
            bound_log_p_change = torch.log(self.action_scale * (1 - at.pow(2)) + epsilon).sum(-1)
            # print(f"count nan bound_log_p_change: {torch.isnan(bound_log_p_change).sum()}")
            log_p_min -= bound_log_p_change
            log_p_max -= bound_log_p_change
            # print(f"count nan, logpmax: {torch.isnan(log_p_max).sum()}, logpmin: {torch.isnan(log_p_min).sum()}")       
            return log_p_max, log_p_min

        if isinstance(state, torch.Tensor):
            debug=False
        else: debug = False
        if debug:
            print(f"-------- in SAC sample ----------")
            print(f"state delta: {state.delta.min().item()}, {state.delta.max().item()}")
        mean, log_std = self.forward(state)
        # original_mean = mean
        if debug:
            print(f"mean delta: {mean.delta.min().item()}, {mean.delta.max().item()}")
        std = log_std.exp()
        if debug:
            print(f"std: {std.item()}")
        # mean# = NN(state), std
        # action# = mean# + N(0, 1) * stds
        # FIXME

        # normal = Normal(torch.zeros(mean.shape).to(mean.device), std)
        zero_mean = torch.zeros_like(mean if isinstance(mean, torch.Tensor) else mean.c)
        one_std = torch.ones_like(std)
        # one_std = std
        normal = Normal(zero_mean, one_std) # Normal(zero_mean, std)
        eps = normal.rsample() # for reparameterization trick (mean + std * N(0,1))
        # eliminate the probability that mean_0 comes from (eps+delta)
        x_t = mean.add(std * eps) # eps ~ N(0, 1)
        original_at = x_t
        if debug:
            print(f"x_t delta: {x_t.delta.min().item()}, {x_t.delta.max().item()}")

        # ? x_t ?= sample from Normal(mean, std)
        # the same as 
        y_t = x_t.tanh() # enforcing action bound
        if debug:
            print(f"y_t delta: {y_t.delta.min().item()}, {y_t.delta.max().item()}")
        action = y_t.mul(self.action_scale).add(self.action_bias)
        if debug:
            print(f"action delta: {action.delta.min().item()}, {action.delta.max().item()}")
            exit(0)
        log_prob = normal.log_prob(eps)
        # if mode == "preRADIAL":
        #     print(f"eps: {eps}")
        #     print(f"log_prob: {log_prob}")
        #     exit(0)
        # Enforcing Action Bound: https://arxiv.org/pdf/1812.05905.pdf
        # I am using y_t center to represent the gradient
        # TODO: use a piece-wise linear function to approximate tanh
        if isinstance(y_t, torch.Tensor):
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        else:
            log_prob -= torch.log(self.action_scale * (1 - y_t.c.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True) # sum
        mean = mean.tanh().mul(self.action_scale).add(self.action_bias)
        if mode == "preRADIAL":
            return action, log_prob, mean, original_at
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(SymbolicGaussianPolicy, self).to(device)


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            high = np.array(action_space.high)
            low = np.array(action_space.low)
            self.action_scale = torch.FloatTensor((high - low) / 2.0)
            self.action_bias = torch.FloatTensor((high + low) / 2.0)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound      
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True) # sum
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
