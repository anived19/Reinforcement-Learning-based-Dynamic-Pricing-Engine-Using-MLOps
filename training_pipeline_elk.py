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
from datetime import datetime, timezone
import json
import structlog
from typing import Dict,  Any
from elasticsearch import Elasticsearch
from pricing_env_elk import DynamicPricingEnv, ELKLogger

try:
    es = Elasticsearch("http://localhost:9200")
    ELK_AVAILABLE = es.ping()
except Exception:
    ELK_AVAILABLE = False


class ELKMLflowCallback(BaseCallback):
    """Enhanced callback to log metrics to both MLflow and ELK Stack during training"""
    
    def __init__(self, eval_env, eval_freq=1000, verbose=0, elk_logger=None):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.elk_logger = elk_logger or ELKLogger()
        self.training_run_id = f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _on_step(self) -> bool:
        # Log training metrics every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation episode
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            actions_taken = []
            prices_set = []

            for _ in range(100):  # max ep length
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                actions_taken.append(float(action[0]))
                prices_set.append(info.get("current_price", 0))

                if terminated or truncated:
                    break

            # Calculate additional metrics
            price_volatility = np.std(prices_set) if len(prices_set) > 1 else 0
            avg_action = np.mean(actions_taken) if actions_taken else 0
            
            # Log to MLflow
            mlflow_metrics = {
                "eval/episode_reward": episode_reward,
                "eval/episode_length": episode_length,
                "eval/avg_price": np.mean(prices_set) if prices_set else 0,
                "eval/price_volatility": price_volatility,
                "eval/total_profit": info.get("total_profit", 0),
                "training/step": self.n_calls,
                "eval/avg_action": avg_action
            }
            mlflow.log_metrics(mlflow_metrics, step=self.n_calls)

            # Log to ELK Stack
            elk_metrics = {
                "training_run_id": self.training_run_id,
                "evaluation_step": self.n_calls,
                "model_type": self.model.__class__.__name__,
                **mlflow_metrics,
                "actions_distribution": {
                    "mean": avg_action,
                    "std": np.std(actions_taken) if len(actions_taken) > 1 else 0,
                    "min": min(actions_taken) if actions_taken else 0,
                    "max": max(actions_taken) if actions_taken else 0
                },
                "price_distribution": {
                    "mean": np.mean(prices_set) if prices_set else 0,
                    "std": price_volatility,
                    "min": min(prices_set) if prices_set else 0,
                    "max": max(prices_set) if prices_set else 0
                }
            }
            
            self.elk_logger.log_performance_metrics(elk_metrics, "training-evaluation")

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

        return True

class ELKRLTrainer:
    """Enhanced RL training pipeline with ELK Stack and MLflow integration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = "models"
        self.logs_dir = "logs"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize ELK logger
        self.elk_logger = ELKLogger()
        
        # MLflow setup - suppress database creation logs
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("dynamic_pricing_rl_elk")
        
        # Initialize structured logger
        self.logger = structlog.get_logger("rl_trainer")

    def create_environment(self, env_config: Dict = None):
        """Create and wrap the training environment"""
        env_config = env_config or self.config.get("env_config", {})

        # Create base environment
        env = DynamicPricingEnv(env_config)

        # Wrap with Monitor for logging
        env = Monitor(env, self.logs_dir)
        return env

    def train_agent(self, algorithm: str = "PPO", total_timesteps: int = 50000):
        """Train RL agent with specified algorithm and enhanced logging"""

        run_name = f"{algorithm}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        training_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Log training start to ELK Stack
        self.elk_logger.log_performance_metrics({
            "training_session_id": training_session_id,
            "event_subtype": "training_start",
            "algorithm": algorithm,
            "total_timesteps": total_timesteps,
            "config": self.config,
            "run_name": run_name
        }, "training-lifecycle")

        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            # Log experiment parameters
            mlflow_params = {
                "algorithm": algorithm,
                "total_timesteps": total_timesteps,
                "env_config": self.config.get("env_config", {}),
                "training_date": datetime.now().isoformat(),
                "training_session_id": training_session_id
            }
            mlflow.log_params(mlflow_params)

            # Create environments
            train_env = self.create_environment()
            eval_env = self.create_environment()

            # Choose algorithm
            if algorithm == "PPO":
                model_params = {
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2
                }
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    **model_params,
                    verbose=1,
                    tensorboard_log=f"{self.logs_dir}/tensorboard/"
                )

            elif algorithm == "A2C":
                model_params = {
                    "learning_rate": 7e-4,
                    "n_steps": 5,
                    "gamma": 0.99,
                    "gae_lambda": 1.0
                }
                model = A2C(
                    "MlpPolicy",
                    train_env,
                    **model_params,
                    verbose=1,
                    tensorboard_log=f"{self.logs_dir}/tensorboard/"
                )
            else:
                raise ValueError(f"Algorithm {algorithm} not supported")

            # Log model hyperparameters to both MLflow and ELK
            model_hyperparams = model.get_parameters()
            mlflow.log_params(model_hyperparams)
            
            self.elk_logger.log_performance_metrics({
                "training_session_id": training_session_id,
                "event_subtype": "hyperparameters",
                "algorithm": algorithm,
                "model_params": model_params,
                "model_hyperparams": model_hyperparams
            }, "training-config")

            # Set up callbacks
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"{self.models_dir}/{algorithm}/best_model",
                log_path=f"{self.logs_dir}/eval/{algorithm}/",
                eval_freq=5000,
                deterministic=True,
                render=False
            )

            elk_mlflow_callback = ELKMLflowCallback(
                eval_env, 
                eval_freq=2000, 
                elk_logger=self.elk_logger
            )

            # Train model
            self.logger.info(f"Starting training: {algorithm} for {total_timesteps} timesteps")
            
            try:
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=[eval_callback, elk_mlflow_callback],
                    progress_bar=True
                )
                
                training_success = True
                
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                training_success = False
                
                # Log training failure to ELK
                self.elk_logger.log_performance_metrics({
                    "training_session_id": training_session_id,
                    "event_subtype": "training_failed",
                    "error": str(e),
                    "algorithm": algorithm,
                    "completed_timesteps": getattr(elk_mlflow_callback, 'n_calls', 0)
                }, "training-lifecycle")
                
                raise e

            # Save final model
            model_path = f"{self.models_dir}/{algorithm}/{algorithm}_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(f"{self.models_dir}/{algorithm}", exist_ok=True)
            model.save(model_path)

            # Log model artifact
            mlflow.log_artifact(f"{model_path}.zip", f"models/{algorithm}")

            # Final evaluation
            final_metrics = self.evaluate_model(model, eval_env, n_episodes=10)
            mlflow.log_metrics(final_metrics)

            # Log training completion to ELK Stack
            self.elk_logger.log_performance_metrics({
                "training_session_id": training_session_id,
                "event_subtype": "training_complete",
                "algorithm": algorithm,
                "total_timesteps": total_timesteps,
                "model_path": model_path,
                "training_success": training_success,
                "final_metrics": final_metrics
            }, "training-lifecycle")

            self.logger.info(f"Training completed! Model saved to {model_path}")
            self.logger.info("Final evaluation metrics:", **final_metrics)

            return model, final_metrics

    def evaluate_model(self, model, env, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained model performance with ELK logging"""
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        episode_rewards = []
        episode_profits = []
        episode_sales = []
        price_volatilities = []
        action_distributions = []

        # Log evaluation start
        self.elk_logger.log_performance_metrics({
            "evaluation_id": evaluation_id,
            "event_subtype": "evaluation_start",
            "n_episodes": n_episodes,
            "model_type": model.__class__.__name__
        }, "model-evaluation")

        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_actions = []
            episode_prices = []

            while episode_length < 100:  # max episode length
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                episode_actions.append(float(action[0]))
                episode_prices.append(info.get("current_price", 0))

                if terminated or truncated:
                    break

            # Collect episode statistics
            episode_rewards.append(episode_reward)
            episode_profits.append(info.get("total_profit", 0))

            # Get environment statistics
            summary = env.env.get_episode_summary()
            episode_sales.append(summary.get("total_sales", 0))
            price_volatilities.append(summary.get("price_volatility", 0))
            action_distributions.append({
                "mean": np.mean(episode_actions),
                "std": np.std(episode_actions),
                "min": min(episode_actions),
                "max": max(episode_actions)
            })

            # Log individual episode results to ELK
            self.elk_logger.log_performance_metrics({
                "evaluation_id": evaluation_id,
                "event_subtype": "episode_result",
                "episode_number": episode + 1,
                "episode_reward": episode_reward,
                "episode_profit": info.get("total_profit", 0),
                "episode_sales": summary.get("total_sales", 0),
                "episode_length": episode_length,
                "price_volatility": summary.get("price_volatility", 0),
                "action_stats": action_distributions[-1],
                "price_stats": {
                    "mean": np.mean(episode_prices),
                    "std": np.std(episode_prices),
                    "min": min(episode_prices),
                    "max": max(episode_prices)
                }
            }, "model-evaluation")

        # Calculate performance metrics
        metrics = {
            "eval/mean_reward": np.mean(episode_rewards),
            "eval/std_reward": np.std(episode_rewards),
            "eval/mean_profit": np.mean(episode_profits),
            "eval/std_profit": np.std(episode_profits),
            "eval/mean_sales": np.mean(episode_sales),
            "eval/mean_price_volatility": np.mean(price_volatilities),
            "eval/reward_stability": 1.0 / (1.0 + np.std(episode_rewards)),  # higher is better
            "eval/profit_consistency": 1.0 / (1.0 + np.std(episode_profits))
        }

        # Log final evaluation metrics to ELK
        self.elk_logger.log_performance_metrics({
            "evaluation_id": evaluation_id,
            "event_subtype": "evaluation_complete",
            "model_type": model.__class__.__name__,
            **metrics,
            "episode_rewards": episode_rewards,
            "episode_profits": episode_profits,
            "action_distribution_summary": {
                "mean_of_means": np.mean([ad["mean"] for ad in action_distributions]),
                "mean_volatility": np.mean([ad["std"] for ad in action_distributions])
            }
        }, "model-evaluation")

        return metrics

    def compare_algorithms(self, algorithms: list = ["PPO", "A2C"], timesteps: int = 30000):
        """Compare multiple RL algorithms with comprehensive ELK logging"""
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Log comparison start
        self.elk_logger.log_performance_metrics({
            "comparison_id": comparison_id,
            "event_subtype": "comparison_start",
            "algorithms": algorithms,
            "timesteps_per_algorithm": timesteps
        }, "algorithm-comparison")

        results = {}

        for algorithm in algorithms:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training {algorithm}")
            self.logger.info(f"{'='*50}")

            try:
                model, metrics = self.train_agent(algorithm, timesteps)
                results[algorithm] = metrics
                
                # Log individual algorithm results
                self.elk_logger.log_performance_metrics({
                    "comparison_id": comparison_id,
                    "event_subtype": "algorithm_result",
                    "algorithm": algorithm,
                    "training_success": True,
                    **metrics
                }, "algorithm-comparison")
                
            except Exception as e:
                self.logger.error(f"Algorithm {algorithm} failed: {e}")
                results[algorithm] = {"error": str(e)}
                
                # Log algorithm failure
                self.elk_logger.log_performance_metrics({
                    "comparison_id": comparison_id,
                    "event_subtype": "algorithm_failed",
                    "algorithm": algorithm,
                    "error": str(e),
                    "training_success": False
                }, "algorithm-comparison")

        # Print and log comparison results
        self.logger.info(f"\n{'='*50}")
        self.logger.info("ALGORITHM COMPARISON")
        self.logger.info(f"{'='*50}")

        comparison_summary = {}
        for algo, metrics in results.items():
            if "error" not in metrics:
                summary = {
                    "mean_reward": metrics.get("eval/mean_reward", 0),
                    "mean_profit": metrics.get("eval/mean_profit", 0),
                    "reward_stability": metrics.get("eval/reward_stability", 0),
                    "profit_consistency": metrics.get("eval/profit_consistency", 0)
                }
                comparison_summary[algo] = summary
                
                self.logger.info(f"{algo}:")
                self.logger.info(f" Mean Reward: {summary['mean_reward']:.2f}")
                self.logger.info(f" Mean Profit: ${summary['mean_profit']:.2f}")
                self.logger.info(f" Reward Stability: {summary['reward_stability']:.3f}")
                self.logger.info(f" Profit Consistency: {summary['profit_consistency']:.3f}")
                self.logger.info("")

        # Determine best algorithm
        if comparison_summary:
            best_algorithm = max(comparison_summary.keys(), 
                               key=lambda x: comparison_summary[x]["mean_reward"])
            
            self.logger.info(f"Best Algorithm: {best_algorithm}")
        else:
            best_algorithm = None

        # Log final comparison results
        self.elk_logger.log_performance_metrics({
            "comparison_id": comparison_id,
            "event_subtype": "comparison_complete",
            "algorithms_tested": algorithms,
            "results_summary": comparison_summary,
            "best_algorithm": best_algorithm,
            "total_algorithms": len(algorithms),
            "successful_algorithms": len([r for r in results.values() if "error" not in r])
        }, "algorithm-comparison")

        return results

    def test_trained_model(self, model_path: str, n_episodes: int = 5):
        """Test a trained model and visualize performance with ELK logging"""
        test_session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Load model
        try:
            if model_path.endswith(".zip"):
                model = PPO.load(model_path)
            else:
                model = PPO.load(f"{model_path}.zip")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None

        # Log test start
        self.elk_logger.log_performance_metrics({
            "test_session_id": test_session_id,
            "event_subtype": "test_start",
            "model_path": model_path,
            "n_episodes": n_episodes,
            "model_type": model.__class__.__name__
        }, "model-testing")

        # Create test environment
        test_env = self.create_environment()
        self.logger.info(f"Testing model: {model_path}")
        self.logger.info(f"Running {n_episodes} test episodes...\n")

        test_results = []

        for episode in range(n_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            episode_steps = []

            self.logger.info(f" Episode {episode+1}:")

            for step in range(100):  # max steps
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward

                step_data = {
                    "step": step + 1,
                    "action": float(action[0]),
                    "price": info.get("current_price", 0),
                    "reward": reward,
                    "inventory": info.get("inventory", 0),
                    "profit": info.get("total_profit", 0)
                }
                episode_steps.append(step_data)

                if step < 5:  # Print first few steps
                    self.logger.info(f"  Step {step+1}: Price=${step_data['price']:.2f}, "
                                   f"Reward={step_data['reward']:.2f}, "
                                   f"Inventory={step_data['inventory']}")

                if terminated or truncated:
                    break

            episode_summary = test_env.env.get_episode_summary()
            
            episode_result = {
                "episode": episode + 1,
                "episode_reward": episode_reward,
                "final_profit": info.get("total_profit", 0),
                "episode_length": len(episode_steps),
                "episode_summary": episode_summary,
                "first_5_steps": episode_steps[:5]
            }
            
            test_results.append(episode_result)
            
            self.logger.info(f" Episode Reward: {episode_reward:.2f}")
            self.logger.info(f" Final Profit: ${info.get('total_profit', 0):.2f}\n")

            # Log episode to ELK
            self.elk_logger.log_performance_metrics({
                "test_session_id": test_session_id,
                "event_subtype": "test_episode",
                **episode_result
            }, "model-testing")

        # Log test completion
        avg_reward = np.mean([r["episode_reward"] for r in test_results])
        avg_profit = np.mean([r["final_profit"] for r in test_results])
        
        self.elk_logger.log_performance_metrics({
            "test_session_id": test_session_id,
            "event_subtype": "test_complete",
            "model_path": model_path,
            "n_episodes": n_episodes,
            "average_reward": avg_reward,
            "average_profit": avg_profit,
            "test_results_summary": {
                "rewards": [r["episode_reward"] for r in test_results],
                "profits": [r["final_profit"] for r in test_results]
            }
        }, "model-testing")

        # Visualize one episode
        test_env.render()

        return test_env.get_episode_summary()

    def get_elk_dashboard_config(self):
        """Generate Kibana dashboard configuration for the pricing environment"""
        dashboard_config = {
            "version": "7.10.0",
            "objects": [
                {
                    "id": "pricing-environment-overview",
                    "type": "dashboard",
                    "attributes": {
                        "title": "Dynamic Pricing RL - Environment Overview",
                        "description": "Overview of pricing environment performance",
                        "panelsJSON": json.dumps([
                            {
                                "id": "pricing-actions-timeline",
                                "type": "visualization",
                                "gridData": {"x": 0, "y": 0, "w": 48, "h": 15}
                            },
                            {
                                "id": "profit-performance",
                                "type": "visualization", 
                                "gridData": {"x": 0, "y": 15, "w": 24, "h": 15}
                            },
                            {
                                "id": "price-distribution",
                                "type": "visualization",
                                "gridData": {"x": 24, "y": 15, "w": 24, "h": 15}
                            }
                        ])
                    }
                }
            ]
        }
        return dashboard_config

# Configuration for different experiments - FIXED with complete environment parameters
EXPERIMENT_CONFIGS = {
    "basic": {
        "env_config": {
            "base_price": 100.0,
            "price_elasticity": -1.5,
            "episode_length": 100,
            "initial_inventory": 1000,  # Added missing parameter
            "base_demand": 50,          # Added missing parameter
            "demand_variance": 0.2,     # Added missing parameter
            "competitor_reactivity": 0.5,  # Added missing parameter
            "holding_cost": 0.01,       # Added missing parameter
            "stockout_penalty": 10.0    # Added missing parameter
        }
    },
    "high_elasticity": {
        "env_config": {
            "base_price": 100.0,
            "price_elasticity": -2.5,  # more price sensitive customers
            "episode_length": 100,
            "initial_inventory": 1000,
            "base_demand": 50,
            "demand_variance": 0.2,
            "competitor_reactivity": 0.5,
            "holding_cost": 0.01,
            "stockout_penalty": 10.0
        }
    },
    "competitive_market": {
        "env_config": {
            "base_price": 100.0,
            "price_elasticity": -1.5,
            "episode_length": 100,
            "initial_inventory": 1000,
            "base_demand": 50,
            "demand_variance": 0.2,
            "competitor_reactivity": 0.8,  # competitors react more aggressively
            "holding_cost": 0.01,
            "stockout_penalty": 10.0
        }
    }
}

# Example usage
if __name__ == "__main__":
    # Suppress MLflow database creation logs
    import logging
    logging.getLogger('mlflow').setLevel(logging.WARNING)
    logging.getLogger('alembic').setLevel(logging.WARNING)
    
    # Initialize trainer
    trainer = ELKRLTrainer(EXPERIMENT_CONFIGS["basic"])

    # Option 1: Train a single agent
    # print("Training PPO agent....")
    # model, metrics = trainer.train_agent("PPO", total_timesteps=20000)

    # Option 2: Compare multiple algorithms
    results = trainer.compare_algorithms(["PPO", "A2C"], timesteps=15000)

    # Option 3: Test a trained model
    # trainer.test_trained_model("models/PPO/best_model")

    print("Training completed!")
    if ELK_AVAILABLE:
        print("Check ELK Stack:")
        print("- Elasticsearch: http://localhost:9200")
        print("- Kibana: http://localhost:5601")
    else:
        print("Note: ELK Stack was not available during this run")
    print("Check MLflow UI: http://localhost:5000")
    print("Run: python -m mlflow ui --backend-store-uri sqlite:///mlflow.db")