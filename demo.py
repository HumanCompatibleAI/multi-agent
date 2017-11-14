import time

from env import GatheringEnv

env = GatheringEnv(n_agents=2)
env.reset()
env.render()

for _ in range(10000):
    observation_n, reward_n, done_n, info_n = env.step(env.action_space.sample())
    if any(done_n):
        break
    env.render()
    time.sleep(1 / 30)
