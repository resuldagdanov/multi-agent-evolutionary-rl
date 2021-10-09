from torch.multiprocessing import Process, Pipe
from agent import Agent
from sac.utils import pprint, str2bool
from rollout_runner import rollout_worker
import sac.utils as utils
import numpy as np
import torch
import argparse
import random
import time
import threading
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('-num_envs', type=int, help='number of parallel environments to be created', default=2)
parser.add_argument('-popsize', type=int, help='evolutionary population size', default=0)
parser.add_argument('-rollsize', type=int, help='rollout size for agents', default=0)
parser.add_argument('-frames', type=float, help='iteration in millions', default=2)
parser.add_argument('-filter_c', type=int, help='prob multiplier for evo experiences absorbtion into buffer', default=1)
parser.add_argument('-evals', type=int, help='Evals to compute a fitness', default=1)
parser.add_argument('-seed', type=int, help='seed', default=2021)
parser.add_argument('-algo', type=str, help='SAC vs. MADDPG', default='SAC')
parser.add_argument('-savetag', help='saved tag', default='')
parser.add_argument('-gradperstep', type=float, help='gradient steps per frame', default=1.0)
parser.add_argument('-pr', type=float, help='prioritization', default=0.0)
parser.add_argument('-use_gpu', type=str2bool, help='usage of gpu', default=False)
parser.add_argument('-alz', type=str2bool, help='actualize', default=False)
parser.add_argument('-cmd_vel', type=str2bool, help='switch to velocity commands', default=True)
