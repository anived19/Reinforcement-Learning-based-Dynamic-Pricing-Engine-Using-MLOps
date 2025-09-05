import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
import mlflow
import mlflow.pytorch
import os
from datetime import datetime
import json
from typing import Dict, Any
from pricing_environment import DynamicPricingEnv


class MLflowCallback(BaseCallback):
    """Custom callback to log metrics to MLflow during training"""
    
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log training metrics every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation episode
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            
            for _ in range(100):  # Max episode length
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # Log to MLflow
            mlflow.log_metrics({
                "eval/episode_reward": episode_reward,
                "eval/episode_length": episode_length,
                "eval/avg_price": np.mean(self.eval_env.get_wrapper_attr('price_history')),
                "eval/total_profit": info.get('total_profit', 0),
                "training/step": self.n_calls
            }, step=self.n_calls)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
        return True


class RLTrainer:
    """Complete RL training pipeline with experiment tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = "models"
        self.logs_dir = "logs"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # MLflow setup
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("dynamic_pricing_rl")
    
    def create_environment(self, env_config: Dict = None):
        """Create and wrap the training environment"""
        env_config = env_config or self.config.get('env_config', {})
        
        # Create base environment
        env = DynamicPricingEnv(env_config)
        
        # Wrap with Monitor for logging
        env = Monitor(env, self.logs_dir)
        
        return env
    
    def train_agent(self, algorithm: str = "PPO", total_timesteps: int = 50000):
        """Train RL agent with specified algorithm"""
        
        # Start MLflow run
        with mlflow.start_run():
            # Log experiment parameters
            mlflow.log_params({
                "algorithm": algorithm,
                "total_timesteps": total_timesteps,
                "env_config": self.config.get('env_config', {}),
                "training_date": datetime.now().isoformat()
            })
            
            # Create environments
            train_env = self.create_environment()
            eval_env = self.create_environment()
            
            # Choose algorithm
            if algorithm == "PPO":
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=1,
                    tensorboard_log=f"{self.logs_dir}/tensorboard/"
                )
            elif algorithm == "A2C":
                model = A2C(
                    "MlpPolicy",
                    train_env,
                    learning_rate=7e-4,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=1.0,
                    verbose=1,
                    tensorboard_log=f"{self.logs_dir}/tensorboard/"
                )
            else:
                raise ValueError(f"Algorithm {algorithm} not supported")
            
            # Log model hyperparameters
            mlflow.log_params(model.get_parameters())
            
            # Set up callbacks
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"{self.models_dir}/best_model",
                log_path=f"{self.logs_dir}/eval/",
                eval_freq=5000,
                deterministic=True,
                render=False
            )
            
            mlflow_callback = MLflowCallback(eval_env, eval_freq=2000)
            
            # Train the model
            print(f"Training {algorithm} agent for {total_timesteps} timesteps...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, mlflow_callback],
                progress_bar=True
            )
            
            # Save final model
            model_path = f"{self.models_dir}/{algorithm}_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model.save(model_path)
            
            # Log model artifact
            mlflow.log_artifact(f"{model_path}.zip", "models")
            
            # Final evaluation
            final_metrics = self.evaluate_model(model, eval_env, n_episodes=10)
            mlflow.log_metrics(final_metrics)
            
            print(f"Training completed! Model saved to {model_path}")
            print("Final evaluation metrics:", final_metrics)
            
            return model, final_metrics
    
    def evaluate_model(self, model, env, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained model performance"""
        episode_rewards = []
        episode_profits = []
        episode_sales = []
        price_volatilities = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while episode_length < 100:  # Max episode length
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # Collect episode statistics
            episode_rewards.append(episode_reward)
            episode_profits.append(info.get('total_profit', 0))
            
            # Get environment statistics
            summary = env.env.get_episode_summary()
            episode_sales.append(summary.get('total_sales', 0))
            price_volatilities.append(summary.get('price_volatility', 0))
        
        # Calculate performance metrics
        metrics = {
            'eval/mean_reward': np.mean(episode_rewards),
            'eval/std_reward': np.std(episode_rewards),
            'eval/mean_profit': np.mean(episode_profits),
            'eval/mean_sales': np.mean(episode_sales),
            'eval/mean_price_volatility': np.mean(price_volatilities),
            'eval/reward_stability': 1.0 / (1.0 + np.std(episode_rewards))  # Higher is better
        }
        
        return metrics
    
    def compare_algorithms(self, algorithms: list = ["PPO", "A2C"], timesteps: int = 30000):
        """Compare multiple RL algorithms"""
        results = {}
        
        for algorithm in algorithms:
            print(f"\n{'='*50}")
            print(f"Training {algorithm}")
            print(f"{'='*50}")
            
            model, metrics = self.train_agent(algorithm, timesteps)
            results[algorithm] = metrics
            
        # Print comparison
        print(f"\n{'='*50}")
        print("ALGORITHM COMPARISON")
        print(f"{'='*50}")
        
        for algo, metrics in results.items():
            print(f"{algo}:")
            print(f"  Mean Reward: {metrics['eval/mean_reward']:.2f}")
            print(f"  Mean Profit: ${metrics['eval/mean_profit']:.2f}")
            print(f"  Reward Stability: {metrics['eval/reward_stability']:.3f}")
            print()
        
        return results
    
    def test_trained_model(self, model_path: str, n_episodes: int = 5):
        """Test a trained model and visualize performance"""
        
        # Load model
        if model_path.endswith('.zip'):
            model = PPO.load(model_path)
        else:
            model = PPO.load(f"{model_path}.zip")
        
        # Create test environment
        test_env = self.create_environment()
        
        print(f"Testing model: {model_path}")
        print(f"Running {n_episodes} test episodes...\n")
        
        for episode in range(n_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            
            print(f"Episode {episode + 1}:")
            
            for step in range(20):  # Show first 20 steps
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                
                if step < 5:  # Print first few steps
                    print(f"  Step {step+1}: Price=${info['current_price']:.2f}, "
                          f"Reward={reward:.2f}, Inventory={info['inventory']}")
                
                if terminated or truncated:
                    break
            
            print(f"  Episode Reward: {episode_reward:.2f}")
            print(f"  Final Profit: ${info['total_profit']:.2f}\n")
        
        # Visualize one episode
        test_env.render()
        
        return test_env.get_episode_summary()


# Configuration for different experiments
EXPERIMENT_CONFIGS = {
    'basic': {
        'env_config': {
            'base_price': 100.0,
            'price_elasticity': -1.5,
            'episode_length': 100
        }
    },
    'high_elasticity': {
        'env_config': {
            'base_price': 100.0,
            'price_elasticity': -2.5,  # More price-sensitive customers
            'episode_length': 100
        }
    },
    'competitive_market': {
        'env_config': {
            'base_price': 100.0,
            'competitor_reactivity': 0.8,  # Competitors react more aggressively
            'episode_length': 100
        }
    }
}


# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = RLTrainer(EXPERIMENT_CONFIGS['basic'])
    
    # Option 1: Train a single agent
    print("Training PPO agent...")
    model, metrics = trainer.train_agent("PPO", total_timesteps=20000)
    
    # Option 2: Compare multiple algorithms
    # results = trainer.compare_algorithms(["PPO", "A2C"], timesteps=15000)
    
    # Option 3: Test a trained model
    # trainer.test_trained_model("models/best_model")
    
    print("Training completed! Check MLflow UI at: http://localhost:5000")
    print("Run: mlflow ui")