from sac.model import QNetwork, ValueNetwork, Actor
from torch.optim import Adam
import torch
import torch.nn as nn
import sac.utils as utils


class SAC(object):
	def __init__(self, id, num_inputs, action_dim, hidden_size, gamma, critic_lr, actor_lr, tau, alpha, target_update_interval, savetag, foldername, actualize, use_gpu):

		self.num_inputs = num_inputs
		self.action_space = action_dim
		self.gamma = gamma
		self.tau = 0.005
		self.alpha = 0.2
		self.target_update_interval = 1
		self.tracker = utils.Tracker(foldername, ['q_'+savetag, 'qloss_'+savetag, 'value_'+savetag, 'value_loss_'+savetag, 'policy_loss_'+savetag, 'mean_loss_'+savetag, 'std_loss_'+savetag], '.csv',save_iteration=1000, conv_size=1000)
		self.total_update = 0
		self.agent_id = id
		self.actualize = actualize

		self.policy_type = "Gaussian" # TODO: will be removed

		self.critic = QNetwork(self.num_inputs, self.action_space, hidden_size)
		self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
		self.soft_q_criterion = nn.MSELoss()

		self.policy = Actor(self.num_inputs, self.action_space, hidden_size)
		self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

		self.value = ValueNetwork(self.num_inputs, hidden_size)
		self.value_target = ValueNetwork(self.num_inputs, hidden_size)
		self.value_optim = Adam(self.value.parameters(), lr=critic_lr)
		utils.hard_update(self.value_target, self.value)
		self.value_criterion = nn.MSELoss()

		self.policy.cuda()
		self.value.cuda()
		self.value_target.cuda()
		self.critic.cuda()

		#Statistics Tracker
		self.q = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.val = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.value_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.policy_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.mean_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.std_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q_loss = {'min':None, 'max': None, 'mean':None, 'std':None}

	# def select_action(self, state, eval=False):
	#     state = torch.FloatTensor(state).unsqueeze(0)
	#     if eval == False:
	#         self.policy.train()
	#         action, _, _, _, _ = self.policy.evaluate(state)
	#     else:
	#         self.policy.eval()
	#         _, _, _, action, _ = self.policy.evaluate(state)
	#
	#     # action = torch.tanh(action)
	#     action = action.detach().cpu().numpy()
	#     return action[0]

	def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, mask_batch, updates, **ignore):
		# state_batch = torch.FloatTensor(state_batch)
		# next_state_batch = torch.FloatTensor(next_state_batch)
		# action_batch = torch.FloatTensor(action_batch)
		# reward_batch = torch.FloatTensor(reward_batch)
		# mask_batch = torch.FloatTensor(np.float32(mask_batch))

		# reward_batch = reward_batch.unsqueeze(1)  # reward_batch = [batch_size, 1]
		# mask_batch = mask_batch.unsqueeze(1)  # mask_batch = [batch_size, 1]

		"""
		Use two Q-functions to mitigate positive bias in the policy improvement step that is known
		to degrade performance of value based methods. Two Q-functions also significantly speed
		up training, especially on harder task.
		"""
		expected_q1_value, expected_q2_value = self.critic(state_batch, action_batch)
		new_action, log_prob, _, mean, log_std = self.policy.noisy_action(state_batch, return_only_action=False)
		utils.compute_stats(expected_q1_value, self.q)


		if self.policy_type == "Gaussian":
			"""
			Including a separate function approximator for the soft value can stabilize training.
			"""
			expected_value = self.value(state_batch)
			utils.compute_stats(expected_value, self.val)
			target_value = self.value_target(next_state_batch)
			next_q_value = reward_batch + mask_batch * self.gamma * target_value  # Reward Scale * r(st,at) - γV(target)(st+1))
		else:
			"""
			There is no need in principle to include a separate function approximator for the state value.
			We use a target critic network for deterministic policy and eradicate the value value network completely.
			"""
			next_state_action, _, _, _, _, = self.policy.noisy_action(next_state_batch, return_only_action=False)
			target_critic_1, target_critic_2 = self.critic_target(next_state_batch, next_state_action)
			target_critic = torch.min(target_critic_1, target_critic_2)
			next_q_value = reward_batch + mask_batch * self.gamma * target_critic  # Reward Scale * r(st,at) - γQ(target)(st+1)

		"""
		Soft Q-function parameters can be trained to minimize the soft Bellman residual
		JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
		∇JQ = ∇Q(st,at)(Q(st,at) - r(st,at) - γV(target)(st+1))
		"""
		q1_value_loss = self.soft_q_criterion(expected_q1_value, next_q_value.detach())
		q2_value_loss = self.soft_q_criterion(expected_q2_value, next_q_value.detach())
		utils.compute_stats(q1_value_loss, self.q_loss)
		q1_new, q2_new = self.critic(state_batch, new_action)
		expected_new_q_value = torch.min(q1_new, q2_new)

		if self.policy_type == "Gaussian":
			"""
			Including a separate function approximator for the soft value can stabilize training and is convenient to 
			train simultaneously with the other networks
			Update the V towards the min of two Q-functions in order to reduce overestimation bias from function approximation error.
			JV = 𝔼st~D[0.5(V(st) - (𝔼at~π[Qmin(st,at) - log π(at|st)]))^2]
			∇JV = ∇V(st)(V(st) - Q(st,at) + logπ(at|st))
			"""
			next_value = expected_new_q_value - (self.alpha * log_prob)
			value_loss = self.value_criterion(expected_value, next_value.detach())
			utils.compute_stats(value_loss, self.value_loss)
		else:
			pass

		"""
		Reparameterization trick is used to get a low variance estimator
		f(εt;st) = action sampled from the policy
		εt is an input noise vector, sampled from some fixed distribution
		Jπ = 𝔼st∼D,εt∼N[logπ(f(εt;st)|st)−Q(st,f(εt;st))]
		∇Jπ =∇log π + ([∇at log π(at|st) − ∇at Q(st,at)])∇f(εt;st)
		"""
		policy_loss = ((self.alpha * log_prob) - expected_new_q_value)
		utils.compute_stats(policy_loss, self.policy_loss)
		policy_loss = policy_loss.mean()

		# Regularization Loss
		mean_loss = 0.001 * mean.pow(2)
		std_loss = 0.001 * log_std.pow(2)
		utils.compute_stats(mean_loss, self.mean_loss)
		utils.compute_stats(std_loss, self.std_loss)
		mean_loss = mean_loss.mean()
		std_loss = std_loss.mean()


		policy_loss += mean_loss + std_loss

		self.critic_optim.zero_grad()
		q1_value_loss.backward()
		self.critic_optim.step()

		self.critic_optim.zero_grad()
		q2_value_loss.backward()
		self.critic_optim.step()

		if self.policy_type == "Gaussian":
			self.value_optim.zero_grad()
			value_loss.backward()
			self.value_optim.step()
		else:
			value_loss = torch.tensor(0.)

		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		self.total_update += 1
		if self.agent_id == 0:
			self.tracker.update([self.q['mean'], self.q_loss['mean'], self.val['mean'], self.value_loss['mean']
								, self.policy_loss['mean'], self.mean_loss['mean'], self.std_loss['mean']], self.total_update)

		"""
		We update the target weights to match the current value function weights periodically
		Update target parameter after every n(args.target_update_interval) updates
		"""
		if updates % self.target_update_interval == 0 and self.policy_type == "Deterministic":
			utils.soft_update(self.critic_target, self.critic, self.tau)

		elif updates % self.target_update_interval == 0 and self.policy_type == "Gaussian":
			utils.soft_update(self.value_target, self.value, self.tau)
		return value_loss.item(), q1_value_loss.item(), q2_value_loss.item(), policy_loss.item()

	# Save model parameters
	def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, value_path=None):
		if not os.path.exists('models/'):
			os.makedirs('models/')

		if actor_path is None:
			actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
		if critic_path is None:
			critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
		if value_path is None:
			value_path = "models/sac_value_{}_{}".format(env_name, suffix)
		print('Saving models to {}, {} and {}'.format(actor_path, critic_path, value_path))
		torch.save(self.value.state_dict(), value_path)
		torch.save(self.policy.state_dict(), actor_path)
		torch.save(self.critic.state_dict(), critic_path)

	# Load model parameters
	def load_model(self, actor_path, critic_path, value_path):
		print('Loading models from {}, {} and {}'.format(actor_path, critic_path, value_path))
		if actor_path is not None:
			self.policy.load_state_dict(torch.load(actor_path))
		if critic_path is not None:
			self.critic.load_state_dict(torch.load(critic_path))
		if value_path is not None:
			self.value.load_state_dict(torch.load(value_path))