from torch.multiprocessing import Process, Pipe
from parameters import Parameters
from agent import Agent
from sac.utils import pprint
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


if __name__ == "__main__":
    args = Parameters()

    # initiate tracker
    test_tracker = utils.Tracker(args.metric_save, [args.log_fname], '.csv')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
