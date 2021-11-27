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

        print("team : ", _type, "\t", team)

        fitness = [None for _ in range(args.num_envs)]
        frame=0
		
        joint_state = env.reset()
        # joint_state = joint_state.values()
        joint_state = utils.to_tensor(np.array(joint_state))

        rollout_trajectory = [[] for _ in range(args.num_agents)]

        # unless done
        while True:
            # [agent_id, env_id, action]
            # print(_type, team)
            # print("WHILE LOOP")
            if _type == 'pg':
                # print("action pg")
                joint_action = [team[i][0].noisy_action(joint_state[i, :]).detach().numpy() for i in range(args.num_agents)]
                print("pg action:", joint_action)
                # print("pg done")
            else:
                # print("action evo")
                # print("SHAPE:",joint_state[2, :].shape)
                joint_action = [team[i].clean_action(joint_state[i, :]).detach().numpy() for i in range(args.num_agents)]
                print("evo action:", joint_action)
                # print("evo done")

            # print("ACTION RECEIVED")
            # joint_action = [team[i].noisy_action(joint_state[i, :]).detach().numpy() for i in range(args.num_agents)]

			# bound action
            joint_action = np.array(joint_action).clip(-1.0, 1.0)
            # print("array:",joint_action[:, 0, :])
            
            # state --> [agent_id, env_id, obs]; reward --> [agent_id, env_id]; done --> [env_id]; info --> [universe_id]
            # print("action:", np.shape(joint_action))
            next_state, reward, done, global_reward = env.step(actions=joint_action)
            next_state = utils.to_tensor(np.array(next_state))

            newdone = []
            for i in range(args.num_envs):
                if done[i][0] == False:
                    newdone.append(0)
                else:
                    newdone.append(1)

            done = newdone

            # print("STEP DONE", done)
			# grab global reward as fitnesses
            for i, grew in enumerate(global_reward):
                if grew != None:
                    fitness[i] = grew

            # print("FITNESS RECEIVED")
			# push experiences to memory
            if store_transitions:
                
                for agent_id in range(args.num_agents):
                    for env_id in range(args.num_envs):
                        
                        # print("STATE:",np.expand_dims(utils.to_numpy(joint_state)[agent_id, env_id, :], 0)) 
                        # print("NEXT:",next_state[agent_id, env_id, :])
                        # print("ACTION:",joint_action[agent_id, env_id, :])
                        # print("REWARD:", reward, done[env_id])

                        if not done[env_id]:
                            rollout_trajectory[agent_id].append([
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
                    for agent_id, buffer in enumerate(data_bucket):
                        for entry in rollout_trajectory[agent_id]:
                            temp_global_reward = fitness[entry[5]]
                            entry[5] = np.expand_dims(np.array([temp_global_reward], dtype="float32"), 0)
                            buffer.append(entry)

                # break all environments as universe is done            
                break

		# send back id, fitness, total length and shaped fitness using the result pipe
        # print(_type, teams_blueprint, [fitness], frame)
        result_pipe.send([teams_blueprint, [fitness], frame])
