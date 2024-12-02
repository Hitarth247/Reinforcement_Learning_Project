import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(), nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(), nn.Flatten(), nn.Linear(2592, 256), nn.ReLU(), nn.Linear(256, nb_actions), )

    def forward(self, x):
        return self.network(x / 255.)
