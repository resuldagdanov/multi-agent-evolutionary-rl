from torch.multiprocessing import Manager
from sac.sac import SAC
from sac.buffer import Buffer
from sac.utils import hard_update
from sac.model import Actor
from evolution import Evolution
import random
import sys


class Agent:
    def __init__(self, args, _id):
        self.args = args
        self.id = _id

		# initalize evolution module
        self.evolver = Evolution(self.args)

		# initialize population
        self.manager = Manager()
        self.popn = self.manager.list()
        
        for _ in range(self.args.popn_size):
            self.popn.append(Actor(args.state_dim, args.action_dim, args.hidden_size))
            self.popn[-1].eval()
            
        self.algo = SAC(id, args.state_dim, args.action_dim, args.hidden_size, args.gamma, args.critic_lr, args.actor_lr, args.tau, args.alpha, args.target_update_interval, args.savetag, args.aux_save, args.actualize, args.use_gpu)
        
        self.rollout_actor = self.manager.list()
        self.rollout_actor.append(Actor(args.state_dim, args.action_dim, args.hidden_size))

		# initalize buffer
        self.buffer = Buffer(args.buffer_size, buffer_gpu=False, filter_c=args.filter_c)

		# agent metrics
        self.fitnesses = [[] for _ in range(args.popn_size)]

		# best policy
        self.champ_ind = 0
        
    def update_parameters(self):
        td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': -1.0, 'action_high': 1.0}

        self.buffer.referesh()
        if self.buffer.__len__() < 10 * self.args.batch_size:
            return
        self.buffer.tensorify()

        for _ in range(int(self.args.gradperstep * self.buffer.pg_frames)):
            s, ns, a, r, done, global_reward = self.buffer.sample(self.args.batch_size, pr_rew=self.args.priority_rate, pr_global=self.args.priority_rate)
            r *= self.args.reward_scaling
            if self.args.use_gpu:
                s = s.cuda(); ns = ns.cuda(); a = a.cuda(); r = r.cuda(); done = done.cuda(); global_reward = global_reward.cuda()
            self.algo.update_parameters(s, ns, a, r, done, global_reward, 1, **td3args)

        # reset new frame counter to 0
        self.buffer.pg_frames = 0
        
    def evolve(self):

		# one gen of evolution
        if self.args.popn_size > 1:
            
            if self.args.scheme == 'multipoint':
			# make sure that the buffer has been refereshed and tensorified
                buffer_pointer = self.buffer
                
                if buffer_pointer.__len__() < 1000:
                    buffer_pointer.tensorify()
                    
                if random.random() < 0.01:
                    buffer_pointer.tensorify()

				# get sample of states from the buffer
                if buffer_pointer.__len__() < 1000:
                    sample_size = buffer_pointer.__len__()
                else:
                    sample_size = 1000
                    
                if sample_size == 1000 and len(buffer_pointer.sT) < 1000:
                    buffer_pointer.tensorify()
                    
                states, _,_,_,_,_ = buffer_pointer.sample(sample_size, pr_rew=0.0, pr_global=0.0)
                states = states.cpu()
                
            elif self.args.scheme == 'standard':
                states = None
            else:
                sys.exit('Unknown Evo Scheme')

			# net indices of nets that got evaluated this generation (meant for asynchronous evolution workloads)
            net_inds = [i for i in range(len(self.popn))] # hack for a synchronous run

			# evolve
            if self.args.rollout_size > 0:
                self.champ_ind = self.evolver.evolve(self.popn, net_inds, self.fitnesses, [self.rollout_actor[0]], states)
            else:
                self.champ_ind = self.evolver.evolve(self.popn, net_inds, self.fitnesses, [], states)

		# reset fitness metrics
        self.fitnesses = [[] for _ in range(self.args.popn_size)]
        
    def update_rollout_actor(self):
        for actor in self.rollout_actor:
            self.algo.policy.cpu()
            hard_update(actor, self.algo.policy)
            
            if self.args.use_gpu:
                self.algo.policy.cuda()


class TestAgent:
    def __init__(self, args, _id):
        self.args = args
        self.id = _id

		# rollout actor is a template used for MP
        self.manager = Manager()
        self.rollout_actor = self.manager.list()
        
        for _ in range(args.num_agents):
            self.rollout_actor.append(Actor(args.state_dim, args.action_dim, args.hidden_size))
            
    def make_champ_team(self, agents):
        for agent_id, agent in enumerate(agents):
            
            if self.args.popn_size <= 1:
                agent.update_rollout_actor()
                hard_update(self.rollout_actor[agent_id], agent.rollout_actor[0])
                
            else:
                hard_update(self.rollout_actor[agent_id], agent.popn[agent.champ_ind])
