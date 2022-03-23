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
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent)
    from config import DebugConfig
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
        print("agents: ", env.agents)
        print("obs:", obs.keys())
        print("done:", dones)
        print("dead:", env.dead.keys())
    env.terminal()
