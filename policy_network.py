"""
policy_network.py — Minimal stub for MCTS import compatibility.
Not used in Phase II experiments (use_policy_net=False).
"""
import torch.nn as nn

NUM_ACTIONS = 128  # placeholder, not used


class SteanePolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_actions, dropout=0.0):
        super().__init__()
        self.num_actions = num_actions
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def action_to_index(action):
    """Map action tuple to integer index (placeholder)."""
    return 0
