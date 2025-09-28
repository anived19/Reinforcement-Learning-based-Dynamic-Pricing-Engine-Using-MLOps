import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt


class DynamicPricingEnv(gym.Env):
    """
    Custom Environment for Dynamic Pricing using Reinforcement Learning
    
    The agent learns to adjust prices based on market conditions to maximize profit.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        # Default configuration
        default_config={
            'base_price': 100.0,
            'base_cost': 70.0,
            'max_inventory': 1000,
            'initial_inventory': 500,
            'max_demand': 100,
            'price_elasticity': -1.5,  # How sensitive customers are to price changes
            'competitor_reactivity': 0.3,  # How much competitors react to our prices
            'seasonality_amplitude': 0.2,  # Strength of seasonal effects
            'noise_level': 0.1,
            'episode_length': 100  # Number of time steps per episode
        }

        # Merge user config with defaults
        if config is not None:
            default_config.update(config)
        self.config = default_config
        
        # Environment state variables
        self.current_price = self.config['base_price']
        self.inventory = self.config['initial_inventory']
        self.time_step = 0
        self.base_demand = 50.0
        self.competitor_prices = [95.0, 105.0, 98.0]  # 3 competitors
        self.total_revenue = 0.0
        self.total_profit = 0.0
        
        # Action space: continuous price multiplier (-20% to +20%)
        self.action_space = spaces.Box(
            low=-0.2, 
            high=0.2, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space: [normalized_price, inventory_ratio, demand_signal, 
        #                    competitor_avg, time_of_day, day_of_week, seasonality]
        self.observation_space = spaces.Box(
            low=np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
            high=np.array([2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # For tracking performance
        self.episode_rewards = []
        self.price_history = []
        self.sales_history = []
        self.profit_history = []
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset state variables
        self.current_price = self.config['base_price']
        self.inventory = self.config['initial_inventory']
        self.time_step = 0
        self.total_revenue = 0.0
        self.total_profit = 0.0
        
        # Reset competitor prices with some randomness
        base_competitor_price = self.config['base_price']
        self.competitor_prices = [
            base_competitor_price + np.random.normal(0, 5) for _ in range(3)
        ]
        
        # Clear history
        self.price_history = [self.current_price]
        self.sales_history = []
        self.profit_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Apply price change (action is a multiplier)
        price_multiplier = 1.0 + float(action[0])
        new_price = self.current_price * price_multiplier
        
        # Ensure price doesn't go below cost or too high
        new_price = np.clip(new_price, 
                           self.config['base_cost'] * 1.1,  # At least 10% markup
                           self.config['base_price'] * 3.0)  # Max 3x base price
        
        self.current_price = new_price
        
        # Simulate market response
        demand = self._simulate_demand()
        sales = min(demand, self.inventory)  # Can't sell more than inventory
        
        # Update inventory
        self.inventory = max(0, self.inventory - sales)
        
        # Calculate financials
        revenue = self.current_price * sales
        cost = self.config['base_cost'] * sales
        profit = revenue - cost
        
        self.total_revenue += revenue
        self.total_profit += profit
        
        # Calculate reward
        reward = self._calculate_reward(profit, sales, demand)
        
        # Update competitors (they react to our pricing)
        self._update_competitors()
        
        # Restock inventory (simulate supply chain)
        self._restock_inventory()
        
        # Update time
        self.time_step += 1
        
        # Check if episode is done
        terminated = self.time_step >= self.config['episode_length']
        truncated = self.inventory <= 0  # Out of stock
        
        # Store history
        self.price_history.append(self.current_price)
        self.sales_history.append(sales)
        self.profit_history.append(profit)
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _simulate_demand(self) -> float:
        """Simulate customer demand based on current market conditions"""
        
        # Base demand with time-of-day effect
        time_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (self.time_step % 24) / 24)
        base_demand = self.base_demand * time_factor
        
        # Price elasticity effect (higher price = lower demand)
        price_ratio = self.current_price / self.config['base_price']
        price_effect = price_ratio ** self.config['price_elasticity']
        
        # Competitor effect (if competitors are cheaper, our demand decreases)
        avg_competitor_price = np.mean(self.competitor_prices)
        competitor_ratio = self.current_price / avg_competitor_price
        competitor_effect = competitor_ratio ** -0.8  # Negative effect of being expensive
        
        # Seasonality (weekly and daily patterns)
        day_of_week = (self.time_step // 24) % 7
        weekly_seasonality = 1.0 + self.config['seasonality_amplitude'] * np.sin(2 * np.pi * day_of_week / 7)
        
        # Random noise
        noise = 1.0 + np.random.normal(0, self.config['noise_level'])
        
        # Combine all effects
        demand = base_demand * price_effect * competitor_effect * weekly_seasonality * noise
        demand = max(0, demand)  # Demand can't be negative
        
        return demand
    
    def _calculate_reward(self, profit: float, sales: float, demand: float) -> float:
        """Calculate reward for the agent"""
        
        # Primary reward: profit
        profit_reward = profit / 100.0  # Scale down for training stability
        
        # Bonus for maintaining market share (selling close to demand)
        if demand > 0:
            market_share_bonus = (sales / demand) * 10.0
        else:
            market_share_bonus = 0.0
        
        # Penalty for being too far from base price (avoid extreme prices)
        price_ratio = self.current_price / self.config['base_price']
        if price_ratio < 0.7 or price_ratio > 1.5:
            price_penalty = -20.0 * abs(price_ratio - 1.0)
        else:
            price_penalty = 0.0
        
        # Inventory management bonus/penalty
        inventory_ratio = self.inventory / self.config['max_inventory']
        if inventory_ratio < 0.1:  # Too low inventory
            inventory_penalty = -10.0
        elif inventory_ratio > 0.9:  # Too high inventory
            inventory_penalty = -5.0
        else:
            inventory_penalty = 0.0
        
        total_reward = profit_reward + market_share_bonus + price_penalty + inventory_penalty
        
        return total_reward
    
    def _update_competitors(self):
        """Update competitor prices (they react to our pricing)"""
        for i, competitor_price in enumerate(self.competitor_prices):
            # Some competitors copy us, others have their own strategy
            if i == 0:  # Copycat competitor
                target_price = self.current_price * 0.95  # Try to undercut us
                self.competitor_prices[i] += 0.1 * (target_price - competitor_price)
            elif i == 1:  # Premium competitor
                target_price = max(self.config['base_price'] * 1.2, self.current_price * 1.05)
                self.competitor_prices[i] += 0.05 * (target_price - competitor_price)
            else:  # Random competitor
                self.competitor_prices[i] += np.random.normal(0, 2)
            
            # Keep prices reasonable
            self.competitor_prices[i] = np.clip(
                self.competitor_prices[i], 
                self.config['base_cost'] * 1.05, 
                self.config['base_price'] * 2.5
            )
    
    def _restock_inventory(self):
        """Simulate inventory restocking"""
        # Simple restocking: add some inventory each time step
        restock_amount = min(20, self.config['max_inventory'] - self.inventory)
        self.inventory += restock_amount
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        
        # Normalize values to [0, 1] range for better training
        normalized_price = self.current_price / self.config['base_price']
        inventory_ratio = self.inventory / self.config['max_inventory']
        demand_signal = self._simulate_demand() / self.config['max_demand']
        competitor_avg = np.mean(self.competitor_prices) / self.config['base_price']
        
        # Time features
        hour_of_day = (self.time_step % 24) / 24.0
        day_of_week = ((self.time_step // 24) % 7) / 7.0
        
        # Seasonality
        seasonality = 0.5 + 0.5 * np.sin(2 * np.pi * (self.time_step % 168) / 168)  # Weekly cycle
        
        observation = np.array([
            normalized_price,
            inventory_ratio,
            demand_signal,
            competitor_avg,
            hour_of_day,
            day_of_week,
            seasonality
        ], dtype=np.float32)
        
        return observation
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state"""
        return {
            'current_price': self.current_price,
            'inventory': self.inventory,
            'total_revenue': self.total_revenue,
            'total_profit': self.total_profit,
            'competitor_prices': self.competitor_prices.copy(),
            'time_step': self.time_step
        }
    
    def render(self, mode='human'):
        """Render the environment (optional visualization)"""
        if len(self.price_history) > 1:
            plt.figure(figsize=(12, 8))
            
            # Plot price history
            plt.subplot(2, 2, 1)
            plt.plot(self.price_history, label='Our Price')
            plt.axhline(y=np.mean(self.competitor_prices), color='r', linestyle='--', label='Competitor Avg')
            plt.title('Price History')
            plt.legend()
            
            # Plot sales history
            plt.subplot(2, 2, 2)
            plt.plot(self.sales_history, label='Sales')
            plt.title('Sales History')
            plt.legend()
            
            # Plot profit history
            plt.subplot(2, 2, 3)
            plt.plot(self.profit_history, label='Profit')
            plt.title('Profit History')
            plt.legend()
            
            # Plot cumulative profit
            plt.subplot(2, 2, 4)
            cumulative_profit = np.cumsum(self.profit_history)
            plt.plot(cumulative_profit, label='Cumulative Profit')
            plt.title('Cumulative Profit')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    def get_episode_summary(self) -> Dict:
        """Get summary statistics for the completed episode"""
        if not self.profit_history:
            return {}
        
        return {
            'total_profit': sum(self.profit_history),
            'total_revenue': self.total_revenue,
            'avg_price': np.mean(self.price_history),
            'price_volatility': np.std(self.price_history),
            'total_sales': sum(self.sales_history),
            'avg_profit_per_sale': sum(self.profit_history) / max(sum(self.sales_history), 1),
            'episode_length': len(self.profit_history)
        }


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = DynamicPricingEnv()
    
    # Test the environment
    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial info:", info)
    
    # Take a few random actions
    for i in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Action: {action[0]:.3f} (price change)")
        print(f"  New Price: ${info['current_price']:.2f}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Inventory: {info['inventory']}")
        print(f"  Profit: ${info['total_profit']:.2f}")
        print()
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    # Show episode summary
    summary = env.get_episode_summary()
    print("Episode Summary:", summary)
    
    # Render the environment (optional)
    env.render()