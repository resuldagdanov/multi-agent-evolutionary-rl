from torch.multiprocessing import Manager
from sac.utils import compute_stats
import numpy as np
import random
import torch


class Buffer():
	def __init__(self, capacity, buffer_gpu, filter_c=None):
		self.capacity = capacity; self.buffer_gpu = buffer_gpu; self.filter_c = filter_c
		self.manager = Manager()
		self.tuples = self.manager.list() # temporary shared buffer to get experiences from processes
		self.s = []; self.ns = []; self.a = []; self.r = []; self.done = []; self.global_reward = []

		# temporary tensors that cane be loaded in GPU for fast sampling during gradient updates (updated each gen) --> Faster sampling - no need to cycle experiences in and out of gpu 1000 times
		self.sT = None; self.nsT = None; self.aT = None; self.rT = None; self.doneT = None; self.global_rewardT = None

		self.pg_frames = 0; self.total_frames = 0

		#Priority indices
		self.top_r = None
		self.top_g = None

		#Stats
		self.rstats = {'min': None, 'max': None, 'mean': None, 'std': None}
		self.gstats = {'min': None, 'max': None, 'mean': None, 'std': None}

	def data_filter(self, exp):
		self.s.append(exp[0])
		self.ns.append(exp[1])
		self.a.append(exp[2])
		self.r.append(exp[3])
		self.done.append(exp[4])
		self.global_reward.append(exp[5])
		self.pg_frames += 1
		self.total_frames += 1

	def referesh(self):
		# add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
		for _ in range(len(self.tuples)):
			exp = self.tuples.pop()
			self.data_filter(exp)

		# trim to make the buffer size < capacity
		while self.__len__() > self.capacity:
			self.s.pop(0); self.ns.pop(0); self.a.pop(0); self.r.pop(0); self.done.pop(0); self.global_reward.pop(0)

	def __len__(self):
		return len(self.s)

	def sample(self, batch_size, pr_rew=0.0, pr_global=0.0 ):
		# uniform sampling
		ind = random.sample(range(len(self.sT)), batch_size)

		if pr_global != 0.0 or pr_rew !=0.0:
			# prioritization
			num_r = int(pr_rew * batch_size); num_global = int(pr_global * batch_size)
			ind_r = random.sample(self.top_r, num_r)
			ind_global = random.sample(self.top_g, num_global)

			ind = ind[num_r+num_global:] + ind_r + ind_global

		return self.sT[ind], self.nsT[ind], self.aT[ind], self.rT[ind], self.doneT[ind], self.global_rewardT[ind]

	def tensorify(self):
		self.referesh() # referesh first

		if self.__len__() >1:
			self.sT = torch.tensor(np.vstack(self.s))
			self.nsT = torch.tensor(np.vstack(self.ns))
			self.aT = torch.tensor(np.vstack(self.a))
			self.rT = torch.tensor(np.vstack(self.r))
			self.doneT = torch.tensor(np.vstack(self.done))
			self.global_rewardT = torch.tensor(np.vstack(self.global_reward))

			if self.buffer_gpu:
				self.sT = self.sT.cuda()
				self.nsT = self.nsT.cuda()
				self.aT = self.aT.cuda()
				self.rT = self.rT.cuda()
				self.doneT = self.doneT.cuda()
				self.global_rewardT = self.global_rewardT.cuda()

			# prioritized indices update
			self.top_r = list(np.argsort(np.vstack(self.r), axis=0)[-int(len(self.s)/10):])
			self.top_g = list(np.argsort(np.vstack(self.global_reward), axis=0)[-int(len(self.s) / 10):])

			# update stats
			compute_stats(self.rT, self.rstats)
			compute_stats(self.global_rewardT, self.gstats)

