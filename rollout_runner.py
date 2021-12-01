from numpy.core.numeric import roll
from env_wrapper import MultiWalker
import sac.utils as utils
import numpy as np
import random
import sys
from sac.model import Actor
import torch

# rollout evaluate an agent in a complete game
def rollout_worker(args, _id, _type, task_pipe, result_pipe, data_bucket, models_bucket, store_transitions):
    env = MultiWalker(args=args)
    
    np.random.seed(_id)
    random.seed(_id)
    
    while True:
        # wait until a signal is received  to start rollout
        teams_blueprint = task_pipe.recv()

        # kill yourself
        if teams_blueprint == 'TERMINATE':
            sys.exit(0)
 
        if _type == 'test' or _type == 'pg':
            team = models_bucket
        elif _type == 'evo':
            team = [models_bucket[agent_id][popn_id] for agent_id, popn_id in enumerate(teams_blueprint)]
        
        fitness = [None for _ in range(args.num_envs)]
        frame = 0
		
        joint_state = env.reset()
        joint_state = utils.to_tensor(np.array(joint_state))

        rollout_trajectory = [[] for _ in range(args.num_envs)]

        # unless done
        while True:
            if _type == 'pg':
                joint_action = [team[i][0].noisy_action(joint_state[i, :]).detach().numpy() for i in range(args.num_agents)]
            else:
                joint_action = [team[i].clean_action(joint_state[i, :]).detach().numpy() for i in range(args.num_agents)]

			# bound action
            joint_action = np.array(joint_action).clip(-1.0, 1.0)

            next_state, reward, done, global_reward = env.step(actions=joint_action)
            next_state = utils.to_tensor(np.array(next_state))

            newdone = []
            for i in range(args.num_envs):
                if done[i][0] == False:
                    newdone.append(0)
                else:
                    newdone.append(1)

            done = newdone

			# grab global reward as fitnesses
            for i, grew in enumerate(global_reward):
                if grew != None:
                    fitness[i] = grew

			# push experiences to memory
            if store_transitions:
                # print(done[0])
                for env_id in range(args.num_envs):
                    for agent_id in range(args.num_agents):
                        if not done[env_id]:
                            rollout_trajectory[env_id].append([
                                np.expand_dims(utils.to_numpy(joint_state)[agent_id, env_id, :], 0),
                                np.expand_dims(utils.to_numpy(next_state)[agent_id, env_id, :], 0),
                                np.expand_dims(joint_action[agent_id, env_id, :], 0),
                                np.expand_dims(np.array([reward[agent_id, env_id]], dtype="float32"), 0),
                                np.expand_dims(np.array([done[env_id]], dtype="float32"), 0),
                                env_id, _type])

            joint_state = next_state
            frame += args.num_envs
            
            if sum(done) > 0 and sum(done) != len(done):
                k = None  
			
            # done flag is received
            if sum(done) == len(done):
                # push experiences to main
                if store_transitions:
                    # TODO: for now all networks are fed from only one replay memory buffer
                    for env_id, buffer in enumerate(data_bucket):
                        # print(len(data_bucket[0]), len(data_bucket[1]), len(data_bucket[2]))
                        for entry in rollout_trajectory[env_id]:
                            temp_global_reward = fitness[entry[5]]
                            entry[5] = np.expand_dims(np.array([temp_global_reward], dtype="float32"), 0)
                            buffer.append(entry)

                # break all environments as universe is done            
                break

        # print(fitness)
		# send back id, fitness, total length and shaped fitness using the result pipe
        # for env_id, buffer in enumerate(data_bucket):
        #     print(buffer[0])
        
        result_pipe.send([teams_blueprint, [fitness], frame])