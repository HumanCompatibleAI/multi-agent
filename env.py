import tkinter as tk

import gym
import gym.spaces

import numpy as np


class GatheringEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    scale = 10
    width = 51
    height = 11
    agent_colors = ['red', 'blue']

    def __init__(self, n_agents=1):
        super().__init__()
        self.n_agents = n_agents
        self.root = None
        self.action_space = gym.spaces.MultiDiscrete([[0, 3]] * n_agents)
        self.reset()
        self.done = False

    def _step(self, actions):
        assert len(actions) == self.n_agents
        directions = [
            (0, -1),  # up
            (1, 0),   # right
            (0, 1),   # down
            (-1, 0),  # left
        ]
        actions = [directions[a] for a in actions]
        for i, ((dx, dy), (x, y)) in enumerate(zip(actions, self.agents)):
            self.agents[i] = ((x + dx) % self.width, (y + dy) % self.height)

        return [], 0, self.done, None

    def _reset(self):
        self.food = np.zeros((self.width, self.height), dtype=np.bool)
        mid_x = self.width // 2
        mid_y = self.height // 2
        self.food[mid_x - 2:mid_x + 3, mid_y - 2: mid_y + 3] = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ]).T

        self.agents = [(i, 0) for i in range(self.n_agents)]

    def _render(self, mode='human', close=False):
        canvas_width = self.width * self.scale
        canvas_height = self.height * self.scale

        if self.root is None:
            def on_close():
                self.done = True
                self.root.destroy()
                self.root = None
                self.canavs = None
            self.root = tk.Tk()
            self.root.title('Gathering')
            self.root.protocol('WM_DELETE_WINDOW', on_close)
            self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
            self.canvas.pack()

        if close:
            self.root.destroy()
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
                if self.food[x, y]:
                    fill_cell(x, y, 'green')

        for i, (x, y) in enumerate(self.agents):
            fill_cell(x, y, self.agent_colors[i])

        self.root.update()

    def close(self):
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None

    def __del__(self):
        self.close()
