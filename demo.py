import random
import time

from env import GatheringEnv

from model import Policy

env = GatheringEnv(n_agents=2)
state_n = env.reset()
env.render()

policy = Policy(env.state_size)
policy.load_weights()

for _ in range(10000):
    state_n, reward_n, done_n, info_n = env.step([
        policy.select_action(state_n[0]),
        random.randrange(0, 4),
    ])
    if any(done_n):
        break
    env.render()
    time.sleep(1 / 30)
