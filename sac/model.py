import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_SIG_MAX = 5
LOG_SIG_MIN = -10
epsilon = 1e-6


# initialize policy weights
def weights_init_policy_fn(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
		torch.nn.init.constant_(m.bias, 0)

# initialize value function weights
def weights_init_value_fn(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        self.apply(weights_init_policy_fn)
        
    def clean_action(self, state, return_only_action=True):
        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        mean = self.mean_linear(x)
        if return_only_action: return torch.tanh(mean)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
        
    def noisy_action(self, state, return_only_action=True):
        mean, log_std = self.clean_action(state, return_only_action=False)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)

        if return_only_action: return action

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        # log_prob.clamp(-10, 0)

        return action, log_prob, x_t, mean, log_std
        
    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)
        
        return minimum, maximum, mean


class ValueNetwork(nn.Module):
	def __init__(self, state_dim, hidden_dim):
		super(ValueNetwork, self).__init__()

		self.linear1 = nn.Linear(state_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)

		self.apply(weights_init_value_fn)

	def forward(self, state):
		x = F.elu(self.linear1(state))
		x = F.elu(self.linear2(x))
		x = self.linear3(x)
		return x


class QNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size):
		super(QNetwork, self).__init__()

		# Q1 architecture
		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		# Q2 architecture
		self.linear4 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear5 = nn.Linear(hidden_size, hidden_size)
		self.linear6 = nn.Linear(hidden_size, 1)

		self.apply(weights_init_value_fn)

	def forward(self, state, action):
		x1 = torch.cat([state, action], 1)
		x1 = torch.tanh(self.linear1(x1))
		x1 = torch.tanh(self.linear2(x1))
		x1 = self.linear3(x1)

		x2 = torch.cat([state, action], 1)
		x2 = torch.tanh(self.linear4(x2))
		x2 = torch.tanh(self.linear5(x2))
		x2 = self.linear6(x2)

		return x1, x2
