import numpy as np


class MultiWalker:
    def __init__(self, args):
        from pettingzoo.sisl import multiwalker_v7

        self.args = args
        self.num_envs = args.num_envs

        self.action_low = -1.0
        self.action_high = 1.0
        self.action_dim = 4
        self.state_dim = 31

        self.global_reward = [0.0 for _ in range(self.num_envs)]
        self.env_dones = [False for _ in range(self.num_envs)]

        # collection of all envs running in parallel
        self.universe = []

        for _ in range(self.num_envs):
            env = multiwalker_v7.parallel_env(n_walkers=self.args.num_agents, position_noise=1e-3, angle_noise=1e-3,
                                    local_ratio=1.0, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0,
                                    terminate_on_fall=True, remove_on_fall=True, max_cycles=500)
            self.universe.append(env)

    def reset(self):
        self.global_reward = [0.0 for _ in range(self.num_envs)]
        self.env_dones = [False for _ in range(self.num_envs)]

        joint_obs = []
        for env in self.universe:
            obs = env.reset()

            # pettingzoo returns dictionary of observations as each key being a different agent
            obs = list(obs.values())

            joint_obs.append(obs)

        # 2D numpy array -> [agent_id, env_id, :]
        joint_obs = np.stack(joint_obs, axis=1)

        # required for some environments being done
        self.dummy_state = obs
        self.dummy_reward = [0.0] * self.args.num_agents

        return joint_obs

    def step(self, actions):
        joint_obs, joint_reward, joint_done, joint_global = [], [], [], []

        for env_id, env in enumerate(self.universe):
            
            # if this particular env instance in universe is already done:
            if self.env_dones[env_id]:
                joint_obs.append(self.dummy_state)
                joint_reward.append(self.dummy_reward)
                joint_done.append(True)
                joint_global.append(None)
            
            else:
                next_state, reward, done, _ = env.step(actions[:, env_id, :])

                # pettingzoo returns dictionary of observations as each key being a different agent
                next_state = list(next_state.values())
                reward = list(reward.values())
                done = list(done.values())
                
                joint_obs.append(next_state)
                joint_reward.append(reward)
                joint_done.append(done)
                
                self.global_reward[env_id] += sum(reward) / self.args.num_agents
                
                if done:
                    joint_global.append(self.global_reward[env_id])
                    self.env_dones[env_id] = True
                else:
                    joint_global.append(None)

        joint_obs = np.stack(joint_obs, axis=1)
        joint_reward = np.stack(joint_reward, axis=1)
        
        return joint_obs, joint_reward, joint_done, joint_global



