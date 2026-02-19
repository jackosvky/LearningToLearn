import os
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
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

def _task_adaptation_worker(args):
    """Worker function for parallel task adaptation.
    
    Args:
        args: Tuple of (task, meta_state_dict, model_config, inner_steps, n_rounds, inner_lr)
    
    Returns:
        Dict of parameter differences after adaptation
    """
    task, meta_state_dict, model_config, inner_steps, n_rounds, inner_lr = args
    
    # Create a fresh model and load meta-parameters
    model = RNNActorCritic(**model_config)
    model.load_state_dict(meta_state_dict)
    
    # Inner loop: adapt to this specific bandit
    inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
    for _ in range(inner_steps):
        traj = collect_bandit_trajectory(model, task, n_rounds, device='cpu')
        advantages, returns = compute_advantages(traj['rewards'], traj['values'])
        
        policy_loss = -(traj['log_probs'].squeeze() * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(traj['values'].squeeze(), returns)
        loss = policy_loss + 0.5 * value_loss
        
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step()
    
    # Compute parameter difference
    param_diff = {}
    for name, param in model.named_parameters():
        param_diff[name] = (param.data - meta_state_dict[name]).clone()
    
    return param_diff

def reptile_bandit_train_parallel(model, n_outer_iters, meta_batch_size,
                                   inner_steps, n_rounds, inner_lr, outer_lr, 
                                   n_workers=None, device='cpu'):
    """Parallel Reptile meta-training loop using multiprocessing.
    
    Args:
        model: RNNActorCritic model to train
        n_outer_iters: Number of outer (meta) iterations
        meta_batch_size: Number of tasks per meta-batch
        inner_steps: Number of inner adaptation steps per task
        n_rounds: Number of rounds per episode
        inner_lr: Inner loop learning rate
        outer_lr: Outer loop (meta) learning rate
        n_workers: Number of parallel workers (default: cpu_count() - 1)
        device: Device for meta-model (workers always use CPU)
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"Starting parallel training with {n_workers} workers")
    
    # Store model configuration for worker initialization
    model_config = {
        'input_size': 3,
        'hidden_size': model.hidden_size,
        'action_size': model.action_size
    }
    
    for outer_iter in range(n_outer_iters):
        # Save meta-parameters
        meta_state_dict = {name: p.data.clone() for name, p in model.named_parameters()}
        
        # Sample a batch of bandit tasks
        task_batch = [TwoArmedBandit.sample_task() for _ in range(meta_batch_size)]
        
        # Prepare arguments for parallel workers
        worker_args = [
            (task, meta_state_dict, model_config, inner_steps, n_rounds, inner_lr)
            for task in task_batch
        ]
        
        # Parallel task adaptation
        with Pool(processes=n_workers) as pool:
            param_diffs = pool.map(_task_adaptation_worker, worker_args)
        
        # Accumulate parameter differences
        accumulated_diff = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        for param_diff in param_diffs:
            for name in accumulated_diff:
                accumulated_diff[name] += param_diff[name]
        
        # Reptile meta-update
        for name, p in model.named_parameters():
            p.data.copy_(meta_state_dict[name] + outer_lr * (accumulated_diff[name] / meta_batch_size))
        
        if outer_iter % 500 == 0:
            print(f"Outer iter {outer_iter}/{n_outer_iters}")

def test_model_performance(model, p_values, n_test_episodes=10, n_rounds=50, 
                          inner_steps=0, inner_lr=0.02, device='cpu',
                          return_trajectories=False):
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
        return_trajectories: If True, include full episode sequences and hidden states in results
    
    Returns:
        Dict with performance metrics for each p value.
        If return_trajectories=True, also includes:
            - 'hidden_states': List of hidden state sequences (n_test_episodes, n_rounds, hidden_size)
            - 'episode_rewards': List of full reward sequences per episode
            - 'arm_selections': List of full arm selection sequences per episode
    """
    model.eval()
    results = {}
    
    print("\n" + "="*60)
    print("TESTING MODEL PERFORMANCE")
    print("="*60)
    
    for p in p_values:
        episode_rewards = []
        optimal_arm_frequencies = []
        hidden_state_sequences = [] if return_trajectories else None
        arm_selection_sequences = [] if return_trajectories else None
        full_reward_sequences = [] if return_trajectories else None
        
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
            if return_trajectories:
                # Manually collect trajectory to capture hidden states
                hidden_states = []
                rewards = []
                actions = []
                
                hidden = None
                prev_action_onehot = torch.zeros(1, 1, task.n_arms, device=device)
                prev_reward = torch.zeros(1, 1, 1, device=device)
                
                with torch.no_grad():
                    for t in range(n_rounds):
                        rnn_input = torch.cat([prev_action_onehot, prev_reward], dim=-1)
                        policy_logits, value, hidden = model(rnn_input, hidden)
                        
                        action_dist = torch.distributions.Categorical(logits=policy_logits.squeeze(1))
                        action = action_dist.sample()
                        reward = task.step(action.item())
                        
                        hidden_states.append(hidden.detach().clone())
                        rewards.append(reward)
                        actions.append(action.item())
                        
                        prev_action_onehot = torch.nn.functional.one_hot(action, task.n_arms).float().unsqueeze(1)
                        prev_reward = torch.tensor([[[reward]]], dtype=torch.float32, device=device)
                
                traj = {
                    'rewards': torch.tensor(rewards, dtype=torch.float32, device=device),
                    'actions': actions,
                    'hidden_states': hidden_states
                }
            else:
                with torch.no_grad():
                    traj = collect_bandit_trajectory(model, task, n_rounds, device)
            
            # Calculate metrics
            total_reward = traj['rewards'].sum().item()
            optimal_choices = sum(1 for a in traj['actions'] if a == optimal_arm)
            optimal_freq = optimal_choices / n_rounds
            
            episode_rewards.append(total_reward)
            optimal_arm_frequencies.append(optimal_freq)
            
            # Store trajectories if requested
            if return_trajectories:
                hidden_state_sequences.append(torch.stack(traj['hidden_states']))  # (n_rounds, 1, hidden_size)
                arm_selection_sequences.append(traj['actions'])
                full_reward_sequences.append(traj['rewards'].cpu().numpy())
            
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
        
        # Add trajectory data if requested
        if return_trajectories:
            results[p]['hidden_states'] = hidden_state_sequences  # List of (n_rounds, 1, hidden_size)
            results[p]['arm_selections'] = arm_selection_sequences  # List of action sequences
            results[p]['episode_rewards'] = full_reward_sequences  # List of reward sequences
        
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



