import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.network = nn.Sequential(
          nn.Linear(state_size, 32),
          #nn.BatchNorm1d(32)
          nn.ReLU(),
          nn.Linear(32, 64),
          #nn.BatchNorm1d(64)
          nn.ReLU(),
          nn.Linear(64, 64),
          #nn.BatchNorm1d(64)
          nn.ReLU(),
          nn.Linear(64, action_size)
        )

        self.network.apply(init_weights)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.network(state)
