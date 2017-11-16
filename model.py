import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Policy(nn.Module):
    def __init__(self, observation_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(observation_size, 128)
        self.affine2 = nn.Linear(128, 8)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)

    def save_weights(self):
        torch.save(self.state_dict(), 'model.pkl')

    def load_weights(self):
        self.load_state_dict(torch.load('model.pkl'))

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self(Variable(state))
        action = probs.multinomial()
        self.saved_actions.append(action)
        return action.data[0, 0]
