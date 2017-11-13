import time

from env import GatheringEnv

env = GatheringEnv(n_agents=2)
env.reset()
env.render()

for _ in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        break
    env.render()
    time.sleep(1 / 30)
