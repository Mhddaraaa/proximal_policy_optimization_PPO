import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """
    A deep neural network for approximating Q-values.
    """

    def __init__(self, n_state, action_dim, continuous, n_hidden=64):
        """
        Initialize the network.

        Args:
            n_state (int): Dimension of the state space.
            n_action (int): Number of possible actions.
            n_hidden (int): Number of hidden units in the layers.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_state, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden * 4),
            nn.ReLU(),
            nn.Linear(n_hidden * 4, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim)) if continuous else nn.Identity()
        # Initialize the weights of the network
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the network using Kaiming normal initialization.
        which works best with ReLU activation.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (Tensor): Input state.

        Returns:
            Tensor: Output Q-values for each action.
        """
        return self.model(state), self.log_std


class ValueNetwork(nn.Module):
    """
    A deep neural network for approximating Q-values.
    """

    def __init__(self, n_state, n_hidden=64):
        """
        Initialize the network.

        Args:
            n_state (int): Dimension of the state space.
            n_action (int): Number of possible actions.
            n_hidden (int): Number of hidden units in the layers.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_state, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden * 4),
            nn.ReLU(),
            nn.Linear(n_hidden * 4, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
        # Initialize the weights of the network
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the network using Kaiming normal initialization.
        which works best with ReLU activation.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (Tensor): Input state.

        Returns:
            Tensor: Output Q-values for each action.
        """
        return self.model(state)