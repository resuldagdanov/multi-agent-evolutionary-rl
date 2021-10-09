from env_wrapper import MultiWalker
import sac.utils as utils
import numpy as np
import random
import sys


# rollout evaluate an agent in a complete game
def rollout_worker(args, _id, type, task_pipe, result_pipe, data_bucket, models_bucket, store_transitions):
    env = MultiWalker(args=args)
    
    np.random.seed(_id)
    random.seed(_id)
    
    while True:
        # wait until a signal is received  to start rollout
        teams_blueprint = task_pipe.recv()
        
        # kill yourself
        if teams_blueprint == 'TERMINATE':
            sys.exit(0)
            
        team = [models_bucket[agent_id][popn_id] for agent_id, popn_id in enumerate(teams_blueprint)]
        
        fitness = [None for _ in range(args.num_envs)]
        frame=0
		
        joint_state = env.reset()
        joint_state = joint_state.values()
        joint_state = utils.to_tensor(np.array(joint_state))

        rollout_trajectory = [[] for _ in range(args.num_agents)]

        # unless done
        while True:
            # [agent_id, env_id, action]
            joint_action = [team[i][0].noisy_action(joint_state[i, :]).detach().numpy() for i in range(args.num_agents)]

			# bound action
            joint_action = np.array(joint_action).clip(-1.0, 1.0)

            # state --> [agent_id, env_id, obs]; reward --> [agent_id, env_id]; done --> [env_id]; info --> [universe_id]
            next_state, reward, done, global_reward = env.step(actions=joint_action)
            next_state = utils.to_tensor(np.array(next_state))

			# grab global reward as fitnesses
            for i, grew in enumerate(global_reward):
                if grew != None:
                    fitness[i] = grew

			# push experiences to memory
            if store_transitions:
                
                for agent_id in range(args.num_agents):
                    for env_id in range(args.num_envs):
                        
                        if not done[env_id]:
                            rollout_trajectory[agent_id].append([
                                np.expand_dims(utils.to_numpy(joint_state)[agent_id, env_id, :], 0),
                                np.expand_dims(utils.to_numpy(next_state)[agent_id, env_id, :], 0),
                                np.expand_dims(joint_action[agent_id, env_id, :], 0),
                                np.expand_dims(np.array([reward[agent_id, env_id]], dtype="float32"), 0),
                                np.expand_dims(np.array([done[env_id]], dtype="float32"), 0),
                                env_id, type])

			joint_state = next_state
			frame += args.num_envs
            
            if sum(done) > 0 and sum(done) != len(done):
                k = None

			# done flag is received
            if sum(done) == len(done):
				
                # push experiences to main
                if store_transitions:

                    # TODO: for now all networks are fed from only one replay memory buffer
                    for heap in rollout_trajectory:
                        for entry in heap:
                            
                            # fifth index of entry is environment id
                            temp_global_reward = fitness[entry[5]]
                            
                            entry[5] = np.expand_dims(np.array([temp_global_reward], dtype="float32"), 0)
                            data_bucket[0].append(entry)

                # break all environments as universe is done            
                break

		# send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([teams_blueprint, [fitness], frame])
