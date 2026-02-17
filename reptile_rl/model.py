import torch
import torch.nn as nn


class RNNActorCritic(nn.Module):
    """Vanilla RNN with a policy head (actor) and a value head (critic).

    Input at each timestep: [previous_action_onehot, previous_reward]
    Output: action logits (policy) and state value estimate (critic).
    """

    def __init__(self, input_size: int, hidden_size: int, action_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_size = action_size

        # Vanilla RNN (Elman network)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")

        # Policy head — outputs logits over actions
        self.policy_head = nn.Linear(hidden_size, action_size)

        # Value head — outputs scalar state-value estimate
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        rnn_out, hidden = self.rnn(x, hidden)
        policy_logits = self.policy_head(rnn_out)
        value = self.value_head(rnn_out)
        return policy_logits, value, hidden