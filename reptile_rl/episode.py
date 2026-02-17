import torch
import torch.nn as nn
import torch.nn.functional as F

from reptile_rl.model import RNNActorCritic
from reptile_rl.bandit_task import TwoArmedBandit

def collect_bandit_trajectory(
    model: RNNActorCritic,
    bandit_task: TwoArmedBandit,
    n_rounds: int,
    device: str = "cpu",
) -> dict:
    """Run one episode of n_rounds arm-pulls in a bandit task.

    At each round the RNN receives [previous_action_onehot, previous_reward]
    and outputs a policy distribution and value estimate.  An action is sampled
    from the policy, the bandit returns a reward, and everything is recorded.

    Returns a dict with keys: log_probs, rewards, values, actions, entropies.
    """
    log_probs = []
    rewards = []
    values = []
    actions = []
    entropies = []

    hidden = None
    # First timestep: no previous action or reward
    prev_action_onehot = torch.zeros(1, 1, bandit_task.n_arms, device=device)
    prev_reward = torch.zeros(1, 1, 1, device=device)

    for t in range(n_rounds):
        # Concatenate previous action (one-hot) and previous reward as input
        rnn_input = torch.cat([prev_action_onehot, prev_reward], dim=-1)

        policy_logits, value, hidden = model(rnn_input, hidden)

        # Build action distribution
        action_dist = torch.distributions.Categorical(logits=policy_logits.squeeze(1))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # Pull the arm in the bandit environment
        reward = bandit_task.step(action.item())

        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value.squeeze())
        actions.append(action.item())
        entropies.append(entropy)

        # Prepare input for the next timestep
        prev_action_onehot = F.one_hot(action, bandit_task.n_arms).float().unsqueeze(1)
        prev_reward = torch.tensor([[[reward]]], dtype=torch.float32, device=device)

    return {
        "log_probs": torch.stack(log_probs),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
        "values": torch.stack(values),
        "actions": actions,
        "entropies": torch.stack(entropies),
    }

def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
) -> tuple:
    """Compute discounted returns and advantages.

    For bandits each round is nearly independent so returns are close to
    the immediate rewards, but we still support a discount factor for
    generality.

    Returns (advantages, returns) tensors.
    """
    T = len(rewards)
    returns = torch.zeros_like(rewards)
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    advantages = returns - values.detach().squeeze()
    return advantages, returns