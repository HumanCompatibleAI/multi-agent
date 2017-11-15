import collections
import tkinter as tk

import gym
import gym.envs.registration
import gym.spaces

import numpy as np


class GatheringEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    scale = 10
    width = 31
    height = 11
    agent_colors = ['red', 'blue']

    def __init__(self, n_agents=1):
        super().__init__()
        self.n_agents = n_agents
        self.root = None
        self.state_size = self.width * self.height * 3
        self.action_space = gym.spaces.MultiDiscrete([[0, 3]] * n_agents)
        self.observation_space = gym.spaces.MultiDiscrete([[[0, 1]] * self.state_size] * n_agents)
        self._spec = gym.envs.registration.EnvSpec(**_spec)
        self.reset()
        self.done = False

    def _step(self, action_n):
        assert len(action_n) == self.n_agents
        directions = [
            (0, -1),  # up
            (1, 0),   # right
            (0, 1),   # down
            (-1, 0),  # left
        ]
        action_n = [directions[a] for a in action_n]
        next_locations = [a for a in self.agents]
        next_locations_map = collections.defaultdict(list)
        for i, ((dx, dy), (x, y)) in enumerate(zip(action_n, self.agents)):
            next_ = ((x + dx) % self.width, (y + dy) % self.height)
            next_locations[i] = next_
            next_locations_map[next_].append(i)
        for overlappers in next_locations_map.values():
            if len(overlappers) > 1:
                for i in overlappers:
                    next_locations[i] = self.agents[i]
        self.agents = next_locations

        obs_n = self.state_n
        reward_n = [0 for _ in range(self.n_agents)]
        done_n = [self.done] * self.n_agents
        info_n = [{}] * self.n_agents

        for i, a in enumerate(self.agents):
            if self.food[a]:
                self.food[a] = -15
                reward_n[i] = 1

        self.food = (self.food + self.initial_food).clip(max=1)

        return obs_n, reward_n, done_n, info_n

    @property
    def state_n(self):
        agents = np.zeros_like(self.food)
        for a in self.agents:
            agents[a] = 1
        s = np.array([[
            self.food.clip(min=0),
            np.zeros_like(self.food),
            agents,
        ]]).repeat(self.n_agents, axis=0)
        for i, (x, y) in enumerate(self.agents):
            s[i, 1, x, y] = 1
            s[i, 2, x, y] = 0
        return s.reshape((self.n_agents, self.state_size))

    def _reset(self):
        self.food = np.zeros((self.width, self.height), dtype=np.int)
        mid_x = self.width // 2
        mid_y = self.height // 2
        self.food[mid_x - 2:mid_x + 3, mid_y - 2: mid_y + 3] = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ]).T
        self.initial_food = self.food.copy()

        self.agents = [(i, 0) for i in range(self.n_agents)]

        return self.state_n

    def _close_view(self):
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
        self.done = True

    def _render(self, mode='human', close=False):
        canvas_width = self.width * self.scale
        canvas_height = self.height * self.scale

        if self.root is None:
            self.root = tk.Tk()
            self.root.title('Gathering')
            self.root.protocol('WM_DELETE_WINDOW', self._close_view)
            self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
            self.canvas.pack()

        if close:
            self._close_view()
            return

        self.canvas.delete(tk.ALL)
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill='black')

        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * self.scale,
                y * self.scale,
                (x + 1) * self.scale,
                (y + 1) * self.scale,
                fill=color,
            )

        for x in range(self.width):
            for y in range(self.height):
                if self.food[x, y] == 1:
                    fill_cell(x, y, 'green')

        for i, (x, y) in enumerate(self.agents):
            fill_cell(x, y, self.agent_colors[i])

        self.root.update()

    def _close(self):
        self._close_view()

    def __del__(self):
        self.close()


_spec = {
    'id': 'Gathering-v0',
    'entry_point': GatheringEnv,
    'reward_threshold': 100,
}


gym.envs.registration.register(**_spec)
