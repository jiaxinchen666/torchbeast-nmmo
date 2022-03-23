import random

import nmmo


def print_config(config):
    for attr in dir(config):
        if not attr.startswith('__'):
            print('{}: {}'.format(attr, getattr(config, attr)))


def random_action():
    return {
        nmmo.action.Move: {
            nmmo.action.Direction: random.choice([0, 1, 2, 3])
        }
    }


if __name__ == "__main__":
    from .config import DebugConfig
    env = nmmo.Env(config=DebugConfig())
    obs = env.reset()
    print(env.agents)
    horizon = 20
    t = 0
    while True:
        actions = {agent_id: random_action() for agent_id in env.agents}
        obs, rewards, dones, infos = env.step(actions)
        t += 1
        if t >= horizon:
            break
        print(f"agents: {env.agents}, dead: {env.dead.keys()}")
        print(f"obs: {obs.keys()}, dones: {dones.keys()}")
    env.terminal()
