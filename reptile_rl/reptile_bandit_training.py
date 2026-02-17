import torch
import numpy as np
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

def test_model_performance(model, p_values, n_test_episodes=10, n_rounds=50, 
                          inner_steps=0, inner_lr=0.02, device='cpu'):
    """
    Test the model's performance on bandit tasks with different p values.
    
    Args:
        model: Trained RNNActorCritic model
        p_values: List of p values to test (e.g., [0.1, 0.3, 0.5, 0.7, 0.9])
        n_test_episodes: Number of test episodes per p value
        n_rounds: Number of rounds per episode
        inner_steps: Number of gradient steps for task adaptation (0 for zero-shot)
        inner_lr: Learning rate for inner adaptation steps
        device: Device to run on
    
    Returns:
        Dict with performance metrics for each p value
    """
    model.eval()
    results = {}
    
    print("\n" + "="*60)
    print("TESTING MODEL PERFORMANCE")
    print("="*60)
    
    for p in p_values:
        episode_rewards = []
        optimal_arm_frequencies = []
        
        for episode in range(n_test_episodes):
            # Create bandit task with specific p
            task = TwoArmedBandit(p)
            optimal_arm = task.optimal_arm()
            
            # Save model state for adaptation
            if inner_steps > 0:
                saved_weights = {name: param.data.clone() 
                               for name, param in model.named_parameters()}
                
                # Inner loop adaptation
                model.train()
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
                
                model.eval()
            
            # Collect test trajectory
            with torch.no_grad():
                traj = collect_bandit_trajectory(model, task, n_rounds, device)
            
            # Calculate metrics
            total_reward = traj['rewards'].sum().item()
            optimal_choices = sum(1 for a in traj['actions'] if a == optimal_arm)
            optimal_freq = optimal_choices / n_rounds
            
            episode_rewards.append(total_reward)
            optimal_arm_frequencies.append(optimal_freq)
            
            # Restore model weights if we did adaptation
            if inner_steps > 0:
                for name, param in model.named_parameters():
                    param.data.copy_(saved_weights[name])
        
        # Compute statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_optimal_freq = np.mean(optimal_arm_frequencies)
        std_optimal_freq = np.std(optimal_arm_frequencies)
        
        results[p] = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_optimal_freq': avg_optimal_freq,
            'std_optimal_freq': std_optimal_freq,
            'max_possible_reward': n_rounds * max(p, 1-p)
        }
        
        # Print results
        print(f"\np = {p:.2f} (Optimal arm: {task.optimal_arm()}, Expected reward: {max(p, 1-p):.2f})")
        print(f"  Avg Reward: {avg_reward:.2f} ± {std_reward:.2f} (out of {n_rounds})")
        print(f"  Avg Optimal Arm Selection: {avg_optimal_freq*100:.1f}% ± {std_optimal_freq*100:.1f}%")
        print(f"  Efficiency: {(avg_reward / results[p]['max_possible_reward'])*100:.1f}%")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    overall_avg_reward = np.mean([r['avg_reward'] for r in results.values()])
    overall_avg_optimal_freq = np.mean([r['avg_optimal_freq'] for r in results.values()])
    print(f"Overall Average Reward: {overall_avg_reward:.2f}")
    print(f"Overall Optimal Arm Selection: {overall_avg_optimal_freq*100:.1f}%")
    print("="*60 + "\n")
    
    return results

# Instantiate and train
model = RNNActorCritic(input_size=3, hidden_size=32, action_size=2)  # input = 2 (action) + 1 (reward)
reptile_bandit_train(model, n_outer_iters=10000, meta_batch_size=10,
                     inner_steps=5, n_rounds=50, inner_lr=0.02, outer_lr=0.1)

# Test the trained model
test_p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
test_results = test_model_performance(
    model, 
    p_values=test_p_values,
    n_test_episodes=10,
    n_rounds=50,
    inner_steps=0,  # Test zero-shot performance (set to 5 for few-shot adaptation)
    inner_lr=0.02
)

print("\nTest complete! You can also test with adaptation by setting inner_steps > 0")
