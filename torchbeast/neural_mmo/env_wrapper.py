import nmmo
import numpy as np
from gym import Env, Wrapper, spaces


class FeatureParser:
    def __init__(self, feas_dim):

        self.map_size = feas_dim[0]

    def parse(self, obs):

        frame_list = {}

        for entity in obs.keys():

            obs_agents = obs[entity]["Entity"]["Continuous"]
            agents_frame = obs_agents.reshape(-1) / np.linalg.norm(
                obs_agents.reshape(-1))

            obs_map = obs[entity]["Tile"]["Continuous"]
            local_map = np.zeros((1, self.map_size, self.map_size),
                                 dtype=np.float32)
            agent_map = np.zeros((1, self.map_size, self.map_size),
                                 dtype=np.float32)

            init_R = obs_map[0][2]
            init_C = obs_map[0][3]

            for line in obs_map:
                local_map[0][int(line[2] - init_R),
                             int(line[3] - init_C)] = line[1]
                if line[0] != 0:
                    agent_map[0][int(line[2] - init_R),
                                 int(line[3] - init_C)] = 1

            map_frame = np.concatenate([local_map, agent_map])

            frame_list[entity] = {
                "agents_frame": agents_frame,
                "map_frame": map_frame
            }

        return frame_list


class RewardParser:
    def parse(self, pre_obs, obs):

        reward_list = {}

        for entity in obs.keys():

            pre_feas = pre_obs[entity]["Entity"]["Continuous"][0]
            feas = obs[entity]["Entity"]["Continuous"][0]

            reward = 0

            # _, AgentID, _, last_level, _, last_R, last_C, last_damage, last_timealive, last_food, last_water, last_health, last_freeze = self_feas[entity]
            # _, AgentID, _,      level, _,      R,      C,      damage,      timealive,      food,      water,      health,      freeze = feas[entity]

            last_timealive = pre_feas[8]
            timealive = feas[8]

            if timealive > last_timealive:
                reward = 1

            reward_list[entity] = reward

        return reward_list


class NMMOWrapper(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.feature_parser = FeatureParser(feas_dim=[15])
        self.reward_parser = RewardParser()
        self.observation_space = spaces.Dict({
            "agents_frame":
            spaces.Box(low=0, high=255, shape=(1300, ), dtype=np.float32),
            "map_frame":
            spaces.Box(low=0, high=255, shape=(2, 15, 15), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(8)
        self._dead_agents = []
        self.agents = []
        self._dummy_feature = {
            key: np.zeros(shape=val.shape, dtype=val.dtype)
            for key, val in self.observation_space.items()
        }

    def reset(self):
        raw_obs = super().reset()
        obs = self.feature_parser.parse(raw_obs)
        self._dead_agents.clear()
        self.agents = list(self.env.agents)  # initial agents
        self._prev_raw_obs = raw_obs
        return obs

    def step(self, actions):
        decisions = self._parse_action(actions)
        raw_obs, reward, done, info = super().step(decisions)
        obs = self.feature_parser.parse(raw_obs)
        reward = self.reward_parser.parse(self._prev_raw_obs, raw_obs)

        for agent_id, d in done.items():
            if d:
                self._dead_agents.append(agent_id)

        for agent_id in self._dead_agents:
            obs[agent_id] = self._dummy_feature
            reward[agent_id] = 0
            done[agent_id] = True

        self._prev_raw_obs = raw_obs
        return obs, reward, done, info

    def _parse_action(self, actions):
        """
        decisions = {
            action.Attack: {
                action.Style: 0,
                action.Target: 1
            },
            action.Move: {
                action.Direction: 1
            }
        }
        """
        decisions = {}
        for agent_id, act in actions.items():
            if agent_id in self._dead_agents:
                continue
            if act == 0:
                decisions[agent_id] = {}
            elif 1 <= act <= 3:
                decisions[agent_id] = {
                    nmmo.action.Attack: {
                        nmmo.action.Style: act - 1,
                        nmmo.action.Target: 2
                    }
                }
            elif 4 <= act <= 7:
                decisions[agent_id] = {
                    nmmo.action.Move: {
                        nmmo.action.Direction: act - 4
                    }
                }
            else:
                raise ValueError(f"invalid action: {act}")
        return decisions


if __name__ == "__main__":
    import time
    from .config import DebugConfig
    env = NMMOWrapper(nmmo.Env(config=DebugConfig()))
    for i in range(10):
        start, step = time.time(), 0
        env.reset()
        episode_return = {agent_id: 0 for agent_id in env.agents}
        while True:
            actions = {
                agent_id: env.action_space.sample()
                for agent_id in env.agents
            }
            obs, reward, done, info = env.step(actions)
            step += 1
            for agent_id, rew in reward.items():
                episode_return[agent_id] += rew
            if all(done.values()):
                break
        print(
            f"episode {i}, total step: {step}, episode return: {episode_return}, elapsed: {time.time() - start}"
        )
