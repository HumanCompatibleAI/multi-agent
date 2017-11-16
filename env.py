import collections
import itertools
import tkinter as tk

import gym
import gym.envs.registration
import gym.spaces

import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ROTATE_RIGHT = 4
ROTATE_LEFT = 5
LASER = 6
NOOP = 7


class GatheringEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    scale = 20
    viewbox_width = 10
    viewbox_depth = 20
    padding = max(viewbox_width // 2, viewbox_depth - 1)
    width = 31 + (padding + 1) * 2
    height = 11 + (padding + 1) * 2
    agent_colors = ['red', 'yellow']

    def __init__(self, n_agents=1):
        self.n_agents = n_agents
        self.root = None
        self.state_size = self.viewbox_width * self.viewbox_depth * 4
        self.action_space = gym.spaces.MultiDiscrete([[0, 7]] * n_agents)
        self.observation_space = gym.spaces.MultiDiscrete([[[0, 1]] * self.state_size] * n_agents)
        self._spec = gym.envs.registration.EnvSpec(**_spec)
        self.reset()
        self.done = False

    def _step(self, action_n):
        assert len(action_n) == self.n_agents
        action_n = [NOOP if self.tagged[i] else a for i, a in enumerate(action_n)]
        self.beams[:] = 0
        directions = [
            (0, -1),  # up
            (1, 0),   # right
            (0, 1),   # down
            (-1, 0),  # left
        ] + [(0, 0)] * 4  # other, non-movement actions
        movement_n = [directions[a] for a in action_n]
        next_locations = [a for a in self.agents]
        next_locations_map = collections.defaultdict(list)
        for i, ((dx, dy), (x, y)) in enumerate(zip(movement_n, self.agents)):
            if self.tagged[i]:
                continue
            next_ = ((x + dx), (y + dy))
            if self.walls[next_]:
                next_ = (x, y)
            next_locations[i] = next_
            next_locations_map[next_].append(i)
        for overlappers in next_locations_map.values():
            if len(overlappers) > 1:
                for i in overlappers:
                    next_locations[i] = self.agents[i]
        self.agents = next_locations

        for i, act in enumerate(action_n):
            if act == ROTATE_RIGHT:
                self.orientations[i] = (self.orientations[i] + 1) % 4
            elif act == ROTATE_LEFT:
                self.orientations[i] = (self.orientations[i] - 1) % 4
            elif act == LASER:
                self.beams[self._viewbox_slice(i, 5, 20, offset=1)] = 1

        obs_n = self.state_n
        reward_n = [0 for _ in range(self.n_agents)]
        done_n = [self.done] * self.n_agents
        info_n = [{}] * self.n_agents

        for i, a in enumerate(self.agents):
            if self.tagged[i]:
                continue
            if self.food[a]:
                self.food[a] = -15
                reward_n[i] = 1
            if self.beams[a]:
                self.tagged[i] = True

        self.food = (self.food + self.initial_food).clip(max=1)

        return obs_n, reward_n, done_n, info_n

    def _viewbox_slice(self, agent_index, width, depth, offset=0):
        left = width // 2
        right = left if width % 2 == 0 else left + 1
        x, y = self.agents[agent_index]
        return tuple(itertools.starmap(slice, (
            ((x - left, x + right), (y - offset, y - offset - depth, -1)),      # up
            ((x + offset, x + offset + depth), (y - left, y + right)),          # right
            ((x + left, x - right, -1), (y + offset, y + offset + depth)),      # down
            ((x - offset, x - offset - depth, -1), (y + left, y - right, -1)),  # left
        )[self.orientations[agent_index]]))

    @property
    def state_n(self):
        agents = np.zeros_like(self.food)
        for i, a in enumerate(self.agents):
            if not self.tagged[i]:
                agents[a] = 1

        food = self.food.clip(min=0)
        s = np.zeros((self.n_agents, self.viewbox_width, self.viewbox_depth, 4))
        for i, (orientation, (x, y)) in enumerate(zip(self.orientations, self.agents)):
            if self.tagged[i]:
                continue
            full_state = np.stack([food, np.zeros_like(food), agents, self.walls], axis=-1)
            full_state[x, y, 2] = 0

            xs, ys = self._viewbox_slice(i, self.viewbox_width, self.viewbox_depth)
            observation = full_state[xs, ys, :]

            s[i] = observation if orientation in [UP, DOWN] else observation.transpose(1, 0, 2)

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

        self.walls = np.zeros_like(self.food)
        p = self.padding
        self.walls[p:-p, p] = 1
        self.walls[p:-p, -p - 1] = 1
        self.walls[p, p:-p] = 1
        self.walls[-p - 1, p:-p] = 1

        self.beams = np.zeros_like(self.food)

        self.agents = [(i + self.padding + 1, self.padding + 1) for i in range(self.n_agents)]
        self.orientations = [UP for _ in self.agents]
        self.tagged = [False for _ in self.agents]

        return self.state_n

    def _close_view(self):
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
        self.done = True

    def _render(self, mode='human', close=False):
        if close:
            self._close_view()
            return

        canvas_width = self.width * self.scale
        canvas_height = self.height * self.scale

        if self.root is None:
            self.root = tk.Tk()
            self.root.title('Gathering')
            self.root.protocol('WM_DELETE_WINDOW', self._close_view)
            self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
            self.canvas.pack()

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
                if self.beams[x, y] == 1:
                    fill_cell(x, y, 'yellow')
                if self.food[x, y] == 1:
                    fill_cell(x, y, 'green')
                if self.walls[x, y] == 1:
                    fill_cell(x, y, 'grey')

        for i, (x, y) in enumerate(self.agents):
            if not self.tagged[i]:
                fill_cell(x, y, self.agent_colors[i])

        if False:
            # Debug view: see the first player's viewbox perspective.
            p1_state = self.state_n[0].reshape(self.viewbox_width, self.viewbox_depth, 4)
            for x in range(self.viewbox_width):
                for y in range(self.viewbox_depth):
                    food, me, other, wall = p1_state[x, y]
                    assert sum((food, me, other, wall)) <= 1
                    y_ = self.viewbox_depth - y - 1
                    if food:
                        fill_cell(x, y_, 'green')
                    elif me:
                        fill_cell(x, y_, 'cyan')
                    elif other:
                        fill_cell(x, y_, 'red')
                    elif wall:
                        fill_cell(x, y_, 'gray')
            self.canvas.create_rectangle(
                0,
                0,
                self.viewbox_width * self.scale,
                self.viewbox_depth * self.scale,
                outline='blue',
            )

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
