import torch
from bandit_task import TwoArmedBandit
from model import RNNActorCritic
from episode import collect_bandit_trajectory

def compute_advantages(rewards, values, gamma=0.99):
    """Simple advantage: A_t = R_t - V(s_t) (no bootstrapping for bandits)."""
    # For bandits, each round is essentially independent, so returns ≈ rewards
    returns = rewards.clone()
    advantages = returns - values.detach().squeeze()
    return advantages, returns

def reptile_bandit_train(model, n_outer_iters, meta_batch_size,
                         inner_steps, n_rounds, inner_lr, outer_lr, device='cpu'):
    """Full Reptile meta-training loop over bandit tasks."""
    for outer_iter in range(n_outer_iters):
        # Save meta-parameters
        meta_weights = {name: p.data.clone() for name, p in model.named_parameters()}
        accumulated_diff = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

        # Sample a batch of bandit tasks
        task_batch = [TwoArmedBandit.sample_task() for _ in range(meta_batch_size)]

        for task in task_batch:
            # Reset to meta-parameters
            for name, p in model.named_parameters():
                p.data.copy_(meta_weights[name])

            # Inner loop: adapt to this specific bandit
            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
            for _ in range(inner_steps):
                traj = collect_bandit_trajectory(model, task, n_rounds, device)
                advantages, returns = compute_advantages(traj['rewards'], traj['values'])

                policy_loss = -(traj['log_probs'].squeeze() * advantages).mean()
                value_loss = torch.nn.functional.mse_loss(traj['values'].squeeze(), returns)
                loss = policy_loss + 0.5 * value_loss

                inner_opt.zero_grad()
                loss.backward()
                inner_opt.step()

            # Accumulate parameter difference
            for name, p in model.named_parameters():
                accumulated_diff[name] += (p.data - meta_weights[name])

        # Reptile meta-update
        for name, p in model.named_parameters():
            p.data.copy_(meta_weights[name] + outer_lr * (accumulated_diff[name] / meta_batch_size))

        if outer_iter % 500 == 0:
            print(f"Outer iter {outer_iter}/{n_outer_iters}")

# Instantiate and train
model = RNNActorCritic(input_size=3, hidden_size=32, action_size=2)  # input = 2 (action) + 1 (reward)
reptile_bandit_train(model, n_outer_iters=10000, meta_batch_size=10,
                     inner_steps=5, n_rounds=50, inner_lr=0.02, outer_lr=0.1)
