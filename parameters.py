from sac.utils import str2bool
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-n_envs', type=int, help='number of parallel environments to be created', default=2)
parser.add_argument('-popsize', type=int, help='evolutionary population size', default=3)
parser.add_argument('-rollsize', type=int, help='rollout size for agents', default=3)
parser.add_argument('-evals', type=int, help='evals to compute a fitness', default=1)
parser.add_argument('-frames', type=float, help='iteration in millions', default=2)
parser.add_argument('-filter_c', type=int, help='prob multiplier for evo experiences absorbtion into buffer', default=1)
parser.add_argument('-seed', type=int, help='seed', default=2021)
parser.add_argument('-algo', type=str, help='SAC vs. MADDPG', default='SAC')
parser.add_argument('-savetag', help='saved tag', default='')
parser.add_argument('-gradperstep', type=float, help='gradient steps per frame', default=1.0)
parser.add_argument('-pr', type=float, help='prioritization', default=0.0)
parser.add_argument('-use_gpu', type=str2bool, help='usage of gpu', default=False)
parser.add_argument('-alz', type=str2bool, help='actualize', default=False)
parser.add_argument('-cmd_vel', type=str2bool, help='switch to velocity commands', default=True)


class Parameters:
    def __init__(self):
        self.num_envs = vars(parser.parse_args())['n_envs']
        self.popn_size = vars(parser.parse_args())['popsize']
        self.rollout_size = vars(parser.parse_args())['rollsize']
        self.num_evals = vars(parser.parse_args())['evals']
        self.iterations_bound = int(vars(parser.parse_args())['frames'] * 1000000)
        self.actualize = vars(parser.parse_args())['alz']
        self.priority_rate = vars(parser.parse_args())['pr']
        self.use_gpu = vars(parser.parse_args())['use_gpu']
        self.seed = vars(parser.parse_args())['seed']
        self.gradperstep = vars(parser.parse_args())['gradperstep']
        self.algo_name = vars(parser.parse_args())['algo']
        self.filter_c = vars(parser.parse_args())['filter_c']
        
        # general hyper-parameters
        self.hidden_size = 256
        self.actor_lr = 5e-5
        self.critic_lr = 1e-5
        self.tau = 1e-5
        self.init_w = True
        self.gamma = 0.5 if self.popn_size > 0 else 0.97 # TODO: check whether gamma is really important for population size
        self.batch_size = 512
        self.buffer_size = 100000
        self.reward_scaling = 10.0
        self.action_loss = False
        self.policy_ups_freq = 2
        self.policy_noise = True
        self.policy_noise_clip = 0.4
        self.alpha = 0.2
        self.target_update_interval = 1

        self.state_dim = 33
        self.action_dim = 4
        
        # mutation and cros-over parameters
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.extinction_prob = 0.005
        self.extinction_magnitude = 0.5
        self.weight_clamp = 1000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform
        self.lineage_depth = 10
        self.ccea_reduction = "leniency"
        self.num_anchors = 5
        self.num_elites = 4
        self.num_blends = int(0.15 * self.popn_size)
        
        self.num_test = 10
        self.test_gap = 5

		# save filenames
        self.savetag = vars(parser.parse_args())['savetag'] + \
                           'pop' + str(self.popn_size) + \
                           '_roll' + str(self.rollout_size) + \
                           '_seed' + str(self.seed) + \
		                   ('_sac' if self.is_matd3 else '')

        self.critic_fname = 'critic_' + self.savetag
        self.actor_fname = 'actor_' + self.savetag
        self.log_fname = 'reward_' + self.savetag
        self.best_fname = 'best_' + self.savetag
        
        self.save_foldername = 'results/'
        self.metric_save = self.save_foldername + 'metrics/'
        self.model_save = self.save_foldername + 'models/'
        self.aux_save = self.save_foldername + 'auxiliary/'
		
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)
        if not os.path.exists(self.metric_save):
            os.makedirs(self.metric_save)
        if not os.path.exists(self.model_save):
            os.makedirs(self.model_save)
        if not os.path.exists(self.aux_save):
            os.makedirs(self.aux_save)
