import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe
from sac.utils import pprint, str2bool
from rollout_runner import rollout_worker
from newagent import Agent, TestAgent
from env_wrapper import MultiWalker
import sac.utils as utils
from sac.model import Actor
from sac.buffer import Buffer
import numpy as np
import torch
import argparse
import random
import threading
import time
import os
import sys
from tensorboardX import SummaryWriter

writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('-n_envs', type=int, help='number of parallel environments to be created', default=3)
parser.add_argument('-n_agents', type=int, help='number of agent in each environment', default=3)
parser.add_argument('-popsize', type=int, help='evolutionary population size', default=6)
parser.add_argument('-rollsize', type=int, help='rollout size for agents', default=3) # rollout ?
parser.add_argument('-evals', type=int, help='evals to compute a fitness', default=1)
parser.add_argument('-frames', type=float, help='iteration in millions', default=2)
parser.add_argument('-filter_c', type=int, help='prob multiplier for evo experiences absorbtion into buffer', default=1)
parser.add_argument('-seed', type=int, help='seed', default=2021)
parser.add_argument('-algo', type=str, help='SAC vs. MADDPG', default='SAC')
parser.add_argument('-savetag', help='saved tag', default='')
parser.add_argument('-gradperstep', type=float, help='gradient steps per frame', default=1.0)
parser.add_argument('-pr', type=float, help='prioritization', default=0.0)
parser.add_argument('-use_gpu', type=str2bool, help='usage of gpu', default=False)
parser.add_argument('-alz', type=str2bool, help='actualize', default=False)  # actualize ?
parser.add_argument('-cmd_vel', type=str2bool, help='switch to velocity commands', default=True) # velocity commands ? 


class Parameters:
    def __init__(self):
        self.num_envs = vars(parser.parse_args())['n_envs']
        self.num_agents = vars(parser.parse_args())['n_agents']
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
        self.hidden_size = 64
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        self.tau = 1e-5
        self.init_w = True
        self.gamma = 0.97
        self.batch_size = 32
        self.buffer_size = 100000
        self.reward_scaling = 10.0
        self.action_loss = False
        self.policy_ups_freq = 2
        self.policy_noise = True
        self.policy_noise_clip = 0.4
        self.alpha = 0.2
        self.target_update_interval = 1

        self.state_dim = 31
        self.action_dim = 4
        
        # mutation and cros-over parameters
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.extinction_prob = 0.005
        self.extinction_magnitude = 0.5
        self.weight_clamp = 1000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform
        self.lineage_depth = 10
        self.ccea_reduction = "leniency"  # ccea reduction ? 
        self.num_anchors = 5
        self.num_elites = 2
        self.num_blends = int(0.15 * self.popn_size)
        
        self.num_test = 10
        self.test_gap = 5

		# save filenames
        self.savetag = vars(parser.parse_args())['savetag'] + \
                           'pop' + str(self.popn_size) + \
                           '_roll' + str(self.rollout_size) + \
                           '_seed' + str(self.seed) + \
		                   ('_sac' if self.algo_name else '')

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

class MultiagentEvolution:
    def __init__(self, args):
        self.args = args

        # initialize the multiagent team of agents
        self.buffers = [Buffer(args.buffer_size, buffer_gpu=False, filter_c=args.filter_c) for _ in range(args.num_envs)]
        
        self.agents = [Agent(self.args, _id, self.buffers[int(_id/3)]) for _id in range(args.num_agents*args.num_envs)]

        self.test_agent = TestAgent(self.args, 991)

        # model bucket as references to the corresponding agent's attributes
        self.buffer_bucket = [ag.tuples for ag in self.buffers]
        
        self.popn_bucket = [ag.popn for ag in self.agents]
        self.rollout_bucket = [ag.rollout_actor for ag in self.agents]
        self.test_bucket = self.test_agent.rollout_actor

        # Evolutionary workers
        if self.args.popn_size > 0:
            self.evo_task_pipes = [Pipe() for _ in range(args.popn_size * args.num_evals)]
            self.evo_result_pipes = [Pipe() for _ in range(args.popn_size * args.num_evals)]

            self.evo_workers = [Process(target=rollout_worker, args=(self.args, _id,'evo', self.evo_task_pipes[_id][1], self.evo_result_pipes[_id][0],
                                                    self.buffer_bucket, self.popn_bucket, True))
                                                    for _id in range(args.popn_size * args.num_evals)]

            for worker in self.evo_workers: worker.start()

        # Policy gradient workers
        if self.args.rollout_size > 0:
            self.pg_task_pipes = Pipe()
            self.pg_result_pipes = Pipe()

            self.pg_workers = [
				Process(target=rollout_worker, args=(self.args, 0, 'pg', self.pg_task_pipes[1], self.pg_result_pipes[0],
                                                    self.buffer_bucket, self.rollout_bucket, self.args.rollout_size > 0))]            

            for worker in self.pg_workers: worker.start()

        # test workers
        self.test_task_pipes = Pipe()
        self.test_result_pipes = Pipe()

        self.test_workers = [
            Process(target=rollout_worker, args=(
                                                self.args, 0, 'test', self.test_task_pipes[1], self.test_result_pipes[0],
		                                        None, self.test_bucket, False))]
                                                
        for worker in self.test_workers: worker.start()

        self.best_score = -999
        self.total_frames = 0
        self.gen_frames = 0
        self.test_trace = []

    def make_teams(self, num_agents, popn_size, num_evals):
        temp_inds = []

        for _ in range(num_evals):
            temp_inds += list(range(popn_size))

        all_inds = [temp_inds[:] for _ in range(num_agents)]

        for entry in all_inds:
            random.shuffle(entry)

        teams = [[entry[i] for entry in all_inds] for i in range(popn_size * num_evals)]
        return teams

    def train(self, gen, test_tracker):
        
        teams = self.make_teams(args.num_agents, args.popn_size, args.num_evals)

        # start evolution rollout
        if self.args.popn_size > 0:
            for pipe, team in zip(self.evo_task_pipes, teams):
                pipe[0].send(team)

        # start policy gradient rollout
        if self.args.rollout_size > 0:

            # synch policy gradient actors to its corresponding rollout_bucket
            for agent in self.agents: 
                agent.update_rollout_actor()

            # Start rollouts using the rollout actors
            self.pg_task_pipes[0].send('START') 

            # Policy gradient updates to spin up threads for each agent
            threads = [threading.Thread(target=agent.update_parameters, args=()) for agent in self.agents]

            # start threads
            for thread in threads:
                thread.start()

            # joint threads
            for thread in threads:
                thread.join()

        all_fits = []

        # join evolution rollouts
        if self.args.popn_size > 0:
            for pipe in self.evo_result_pipes:
                entry = pipe[1].recv()
                team = entry[0]
                fitness = entry[1][0]
                frames = entry[2]

                # print(entry)
                for agent_id, popn_id in enumerate(team):
                    self.agents[agent_id*3].fitnesses[popn_id].append(utils.list_mean(fitness))
                    self.agents[agent_id*3+1].fitnesses[popn_id].append(utils.list_mean(fitness))
                    self.agents[agent_id*3+2].fitnesses[popn_id].append(utils.list_mean(fitness))

                # for agent_id, _ in enumerate(self.agents):
                #     print(self.agents[agent_id].fitnesses,"\n")

                all_fits.append(utils.list_mean(fitness))
                self.total_frames += frames
        
        pg_fits = []

        # join policy gradient rollouts
        if self.args.rollout_size > 0:
            entry = self.pg_result_pipes[1].recv()
            pg_fits = entry[1][0]
            self.total_frames += entry[2]
        
        test_fits = []

		# join test rollouts
        if gen % self.args.test_gap == 0:
            entry = self.test_result_pipes[1].recv()
            test_fits = entry[1][0]
            test_tracker.update([utils.list_mean(test_fits)], self.total_frames)
            self.test_trace.append(utils.list_mean(test_fits))
            writer.add_scalar('Test Fits/train', test_fits, self.total_frames)

		# evolution step
        for agent in self.agents:
            agent.evolve()

		# save models periodically
        if gen % 20 == 0:
            print("Models Saved")

            for id, test_actor in enumerate(self.test_agent.rollout_actor):
                torch.save(test_actor.state_dict(), self.args.model_save + str(id) + '_' + self.args.actor_fname)
                
        return all_fits, pg_fits, test_fits        

if __name__ == "__main__":
    args = Parameters()
    # mp.set_start_method('spawn')

    # initiate tracker
    test_tracker = utils.Tracker(args.metric_save, [args.log_fname], '.csv')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    multiagent_evolver = MultiagentEvolution(args)

    time_start = time.time()

    # start training loop
    for gen in range(1, 10000000000):
        popn_fits, pg_fits, test_fits = multiagent_evolver.train(gen, test_tracker)

        print('Ep:/Frames', gen, '/', multiagent_evolver.total_frames, 'Popn stat:', utils.list_stat(popn_fits), 'PG_stat:',
        utils.list_stat(pg_fits), 'Test_trace:', [pprint(i) for i in multiagent_evolver.test_trace[-5:]],
        'FPS:', pprint(multiagent_evolver.total_frames / (time.time() - time_start)))

        if gen % 5 == 0:
            print("\n")
            print('Test_stat:', utils.list_stat(test_fits), 'SAVETAG:  ', args.savetag)
            print('Weight Stats: min/max/average', pprint(multiagent_evolver.test_bucket[0].get_norm_stats()))
            print("\n")
            
        if gen % 10 == 0 and args.rollout_size > 0:
            print("\n")
            print('Q', pprint(multiagent_evolver.agents[0].algo.q))
            print('Q_loss', pprint(multiagent_evolver.agents[0].algo.q_loss))
            print('Policy', pprint(multiagent_evolver.agents[0].algo.policy_loss))
            print('Val', pprint(multiagent_evolver.agents[0].algo.val))
            print('Val_loss', pprint(multiagent_evolver.agents[0].algo.value_loss))
            print('Mean_loss', pprint(multiagent_evolver.agents[0].algo.mean_loss))
            print('Std_loss', pprint(multiagent_evolver.agents[0].algo.std_loss))
            
            print('R_mean:', [agent.buffer.rstats['mean'] for agent in multiagent_evolver.agents])
            print('G_mean:', [agent.buffer.gstats['mean'] for agent in multiagent_evolver.agents])
            
        if multiagent_evolver.total_frames > args.iterations_bound:
            break
    
    # kill all processes
    multiagent_evolver.pg_task_pipes[0].send('TERMINATE')
    multiagent_evolver.test_task_pipes[0].send('TERMINATE')
    
    for p in multiagent_evolver.evo_task_pipes:
        p[0].send('TERMINATE')

    print('Finished Running ', args.savetag)
    sys.exit(0)