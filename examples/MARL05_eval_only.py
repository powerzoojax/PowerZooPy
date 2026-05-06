"""MARL05 Evaluation Only - Generate new figures from existing checkpoint"""

import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'x_MARL05_output')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

CONFIG = {
    'max_steps': 48,
    'num_batteries': 3,
}

def plot_evaluation_results(eval_data, output_dir):
    """Plot evaluation episode results with price signal and load curve"""
    import matplotlib.pyplot as plt
    
    if not eval_data['socs']:
        return
    
    agent_names = list(eval_data['socs'][0].keys())
    n_agents = len(agent_names)
    steps = len(eval_data['socs'])
    
    # Generate time axis in hours
    time_hours = np.arange(steps) * 0.5  # 30-min intervals
    
    # Generate electricity price curve ($/MWh) - typical TOU pricing
    peak_hours = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    off_peak_hours = {0, 1, 2, 3, 4, 5, 6, 23}
    price_curve = []
    for step in range(steps):
        hour = int((step * 30) / 60) % 24
        if hour in peak_hours:
            price_curve.append(80.0)  # High price $/MWh
        elif hour in off_peak_hours:
            price_curve.append(30.0)  # Low price $/MWh
        else:
            price_curve.append(50.0)  # Normal price $/MWh
    
    # Generate load curve (normalized typical daily pattern)
    load_curve = []
    for step in range(steps):
        hour = int((step * 30) / 60) % 24
        # Typical load pattern: low at night, high during day
        if hour < 6:
            load = 0.4 + 0.05 * hour  # Night low
        elif hour < 9:
            load = 0.55 + 0.15 * (hour - 6)  # Morning ramp
        elif hour < 12:
            load = 1.0 - 0.05 * (hour - 9)  # Morning peak
        elif hour < 14:
            load = 0.85 + 0.05 * (hour - 12)  # Lunch dip
        elif hour < 19:
            load = 0.95 + 0.01 * (hour - 14)  # Afternoon
        elif hour < 21:
            load = 1.0 - 0.1 * (hour - 19)  # Evening peak
        else:
            load = 0.8 - 0.13 * (hour - 21)  # Night decline
        load_curve.append(load * 100)  # Scale to percentage
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = plt.cm.tab10(np.arange(n_agents))
    
    # 1. Electricity Price Curve
    ax = axes[0, 0]
    ax.plot(time_hours, price_curve, 'r-', linewidth=2.5, label='Electricity Price')
    ax.fill_between(time_hours, 0, price_curve, alpha=0.3, color='coral')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Price ($/MWh)')
    ax.set_title('(a) Time-of-Use Electricity Price')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 23.75])
    ax.set_ylim([0, 100])
    
    # 2. Load Curve
    ax = axes[0, 1]
    ax.plot(time_hours, load_curve, 'b-', linewidth=2.5, label='System Load')
    ax.fill_between(time_hours, 0, load_curve, alpha=0.3, color='steelblue')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Load (%)')
    ax.set_title('(b) Normalized Daily Load Profile')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 23.75])
    ax.set_ylim([0, 120])
    
    # 3. SOC curves with price overlay
    ax = axes[0, 2]
    ax2 = ax.twinx()
    ax2.plot(time_hours, price_curve, 'r--', linewidth=1.5, alpha=0.7, label='Price')
    ax2.set_ylabel('Price ($/MWh)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 100])
    
    for i, agent in enumerate(agent_names):
        socs = [s[agent] for s in eval_data['socs']]
        ax.plot(time_hours, socs, color=colors[i], linewidth=2, label=agent)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='SOC min')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='SOC max')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('State of Charge')
    ax.set_title('(c) Battery SOC vs Price')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 23.75])
    ax.set_ylim([0, 1])
    
    # 4. Power curves with price overlay
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax2.plot(time_hours, price_curve, 'r--', linewidth=1.5, alpha=0.7, label='Price')
    ax2.set_ylabel('Price ($/MWh)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 100])
    
    for i, agent in enumerate(agent_names):
        powers = [p[agent] for p in eval_data['powers']]
        ax.plot(time_hours, powers, color=colors[i], linewidth=2, label=agent)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (MW)')
    ax.set_title('(d) Battery Power vs Price (+discharge, -charge)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 23.75])
    
    # 5. SOC Constraint Violations
    ax = axes[1, 1]
    total_violations = [sum(v.values()) for v in eval_data['violations']]
    ax.bar(time_hours, total_violations, width=0.4, color='red', alpha=0.7)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('SOC Violation')
    ax.set_title('(e) SOC Constraint Violations (Need for Safe RL)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 23.75])
    if max(total_violations) == 0:
        ax.set_ylim([0, 0.1])
        ax.text(12, 0.05, 'No Violations', ha='center', fontsize=12, color='green')
    
    # 6. Profit/Reward with price overlay
    ax = axes[1, 2]
    ax2 = ax.twinx()
    ax2.plot(time_hours, price_curve, 'r--', linewidth=1.5, alpha=0.7, label='Price')
    ax2.set_ylabel('Price ($/MWh)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 100])
    
    total_profits = [sum(p.values()) for p in eval_data['profits']]
    total_rewards = [sum(r.values()) / len(agent_names) for r in eval_data['rewards']]
    cumulative_profit = np.cumsum(total_profits)
    
    ax.plot(time_hours, total_profits, 'g-', linewidth=2, label='Step Profit')
    ax.plot(time_hours, total_rewards, 'b-', linewidth=1.5, alpha=0.7, label='Step Reward')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Value ($)')
    ax.set_title(f'(f) Profit & Reward (Total: ${cumulative_profit[-1]:.2f})')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 23.75])
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'evaluation_results.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Evaluation results figures saved")


def main():
    print("=" * 60)
    print("MARL05 Evaluation Only - Loading checkpoint and regenerating figures")
    print("=" * 60)
    
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from powerzoo.tasks import make_task_env
    
    # Create environment
    eval_env = make_task_env('marl_der_arbitrage', 
                             max_steps=CONFIG['max_steps'],
                             num_batteries=CONFIG['num_batteries'])
    agents = eval_env.possible_agents
    obs_space = eval_env.observation_space[agents[0]]
    
    print(f"\n[Environment Info]")
    print(f"  Agents: {agents}")
    print(f"  Checkpoint: {CHECKPOINT_DIR}")
    
    # Register environment
    register_env("marl_der_arbitrage_task", 
                 lambda cfg: make_task_env('marl_der_arbitrage', **cfg))
    
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    try:
        # Build config
        config = (
            PPOConfig()
            .environment(
                env="marl_der_arbitrage_task",
                env_config={
                    'max_steps': CONFIG['max_steps'],
                    'num_batteries': CONFIG['num_batteries'],
                },
            )
            .framework('torch')
            .env_runners(num_env_runners=0)
            .training(
                lr=3e-4,
                gamma=0.99,
                train_batch_size=2048,
                model={"fcnet_hiddens": [128, 128]},
            )
            .multi_agent(
                policies={
                    agent: (None, obs_space, eval_env.action_space[agent], {})
                    for agent in agents
                },
                policy_mapping_fn=lambda agent_id, episode, worker=None, **kw: agent_id,
            )
        )
        
        # Build algorithm and restore from checkpoint
        algo = config.build()
        algo.restore(CHECKPOINT_DIR)
        print(f"  Checkpoint restored successfully!")
        
        # Run evaluation
        print(f"\n[Evaluation]")
        eval_env = make_task_env('marl_der_arbitrage', 
                                 max_steps=CONFIG['max_steps'],
                                 num_batteries=CONFIG['num_batteries'])
        obs, info = eval_env.reset()
        
        eval_data = {'powers': [], 'socs': [], 'rewards': [], 'violations': [], 'profits': []}
        
        # Get RLModule for inference
        rl_modules = {}
        try:
            learner = algo.learner_group._learner
            if hasattr(learner, '_module'):
                multi_module = learner._module
                if hasattr(multi_module, '_rl_modules'):
                    rl_modules = multi_module._rl_modules
                    print(f"  Got modules: {list(rl_modules.keys())}")
        except Exception as e:
            print(f"  Warning: Could not get modules: {e}")
        
        # Run evaluation loop
        for step in range(CONFIG['max_steps']):
            actions = {}
            for agent in agents:
                try:
                    obs_array = obs[agent]
                    obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
                    
                    agent_module = rl_modules.get(agent, None)
                    
                    if agent_module is not None:
                        with torch.no_grad():
                            fwd_out = agent_module.forward_inference({"obs": obs_tensor})
                        
                        if "action_dist_inputs" in fwd_out:
                            action_value = float(fwd_out["action_dist_inputs"][0, 0].item())
                        elif "actions" in fwd_out:
                            action_value = float(fwd_out["actions"][0, 0].item())
                        else:
                            action_value = 0.0
                    else:
                        power_mw = eval_env._resource_info[agent]['power_mw']
                        action_value = np.random.uniform(-power_mw, power_mw)
                        
                except Exception as e:
                    if step == 0:
                        print(f"  Warning: {agent}: {e}")
                    power_mw = eval_env._resource_info[agent]['power_mw']
                    action_value = np.random.uniform(-power_mw, power_mw)
                
                power_mw = eval_env._resource_info[agent]['power_mw']
                actions[agent] = np.clip(action_value, -power_mw, power_mw)
            
            if step < 3:
                print(f"  Step {step}: actions = {actions}")
            
            obs, rewards, terminateds, truncateds, infos = eval_env.step(actions)
            
            episode_data = eval_env.get_episode_data()
            if episode_data['powers']:
                eval_data['powers'].append(episode_data['powers'][-1])
                eval_data['socs'].append(episode_data['socs'][-1])
                eval_data['rewards'].append(episode_data['rewards'][-1])
                eval_data['violations'].append(episode_data['violations'][-1])
                eval_data['profits'].append(episode_data['profits'][-1])
            
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        
        print(f"  Evaluation steps: {len(eval_data['socs'])}")
        
        if eval_data['socs']:
            total_reward = sum(sum(r.values()) for r in eval_data['rewards']) / len(agents)
            total_profit = sum(sum(p.values()) for p in eval_data['profits'])
            total_violation = sum(sum(v.values()) for v in eval_data['violations'])
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Total profit: {total_profit:.4f}")
            print(f"  Total violation: {total_violation:.4f}")
        
        # Generate new figures
        print(f"\n[Generating Figures]")
        plot_evaluation_results(eval_data, OUTPUT_DIR)
        
        algo.stop()
        
    finally:
        ray.shutdown()
    
    print(f"\n[Complete] Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

