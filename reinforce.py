# Adapted from https://github.com/pytorch/examples/blob/2dca104/reinforcement_learning/reinforce.py
# Licensed under BSD 3-clause: https://github.com/pytorch/examples/blob/2dca10404443ce3178343c07ba6e22af13efb006/LICENSE

import argparse
import random

from itertools import count

from env import GatheringEnv

from model import Policy

import numpy as np

import torch
import torch.autograd as autograd
import torch.optim as optim


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = GatheringEnv(n_agents=2)
env.seed(args.seed)
torch.manual_seed(args.seed)


policy = Policy(env.state_size)
policy.load_weights()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def finish_episode():
    R = 0  # noqa
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R  # noqa
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]


running_reward = 10
for i_episode in count(1):
    state = env.reset()[0]
    episode_reward = 0
    for t in range(1000):  # Don't infinite loop while learning
        action = policy.select_action(state)
        state_n, reward_n, done_n, _ = env.step([action, random.randrange(0, 4)])
        state = state_n[0]
        reward = reward_n[0]
        done = done_n[0]
        if args.render:
            env.render()
        policy.rewards.append(reward)
        episode_reward += reward
        if done:
            break

    running_reward = running_reward * 0.99 + episode_reward * 0.01
    finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {:5d}\tAverage reward: {:.2f}'.format(
            i_episode, episode_reward, running_reward))
        policy.save_weights()
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode received {} reward!".format(running_reward, episode_reward))
        policy.save_weights()
        break
