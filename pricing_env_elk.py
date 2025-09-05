import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime, timezone
from elasticsearch import Elasticsearch
import structlog

# Configure structured logging for ELK Stack
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

class ELKLogger:
    """Logger for sending structured data to ELK Stack"""
    
    def __init__(self, elasticsearch_host="localhost", elasticsearch_port=9200):
        self.logger = structlog.get_logger("pricing_environment")
        
        # Initialize Elasticsearch client
        try:
            self.es_client = Elasticsearch([f"http://{elasticsearch_host}:{elasticsearch_port}"])
            # Test connection
            if not self.es_client.ping():
                self.logger.warning("Elasticsearch connection failed, using file logging only")
                self.es_client = None
        except Exception as e:
            self.logger.warning(f"Failed to connect to Elasticsearch: {e}, using file logging only")
            self.es_client = None
    
    def log_environment_state(self, env_state: Dict, index_name: str = "pricing-environment"):
        """Log environment state to both file and Elasticsearch"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "@timestamp": timestamp,
            "event_type": "environment_state",
            **env_state
        }
        
        # Log to file (picked up by Filebeat)
        self.logger.info("environment_state", **log_entry)
        
        # Send to Elasticsearch directly
        if self.es_client:
            try:
                self.es_client.index(
                    index=f"{index_name}-{datetime.now().strftime('%Y.%m')}",
                    body=log_entry
                )
            except Exception as e:
                self.logger.error(f"Failed to send to Elasticsearch: {e}")
    
    def log_action_taken(self, action_data: Dict, index_name: str = "pricing-actions"):
        """Log agent actions to ELK Stack"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "@timestamp": timestamp,
            "event_type": "agent_action",
            **action_data
        }
        
        self.logger.info("agent_action", **log_entry)
        
        if self.es_client:
            try:
                self.es_client.index(
                    index=f"{index_name}-{datetime.now().strftime('%Y.%m')}",
                    body=log_entry
                )
            except Exception as e:
                self.logger.error(f"Failed to send to Elasticsearch: {e}")
    
    def log_performance_metrics(self, metrics: Dict, index_name: str = "pricing-performance"):
        """Log performance metrics to ELK Stack"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            "@timestamp": timestamp,
            "event_type": "performance_metrics",
            **metrics
        }
        
        self.logger.info("performance_metrics", **log_entry)
        
        if self.es_client:
            try:
                self.es_client.index(
                    index=f"{index_name}-{datetime.now().strftime('%Y.%m')}",
                    body=log_entry
                )
            except Exception as e:
                self.logger.error(f"Failed to send to Elasticsearch: {e}")

#To make an agent that handles the change in price of items
#the agent learns to adjust price to maximise profit
class DynamicPricingEnv(gym.Env):
    def __init__(self, config: Dict= None):
        super().__init__()

        #default configuration
        self.config = config or {
        "base_price":100.0,
        "base_cost": 70.0,
        "max_inventory": 1000,
        "initial_inventory": 500,
        "max_demand":100,
        "price_elasticity": -1.5, #how sensitive customers are to price changes
        "competitor_reactivity": 0.3, #how much competitors react to our prices
        "seasonality_amplitude":0.2, #strength of seasonal effects
        "noise_level": 0.1,
        "episode_length": 100 #number of time steps per episode
        }

        #environment state variables
        self.current_price= self.config["base_price"]
        self.inventory= self.config["initial_inventory"]
        self.time_step= 0
        self.base_demand= 50.0
        self.competitor_prices= [95.0, 105.0, 98.0] #3 competitors
        self.total_revenue=0.0
        self.total_profit= 0.0
        self.episode_id = None

        # Initialize ELK logger
        self.elk_logger = ELKLogger()

        #action space: continuous price multiplier (-20% to +20%)
        self.action_space= spaces.Box(
            low= -0.2,
            high= 0.2,
            shape=(1,),
            dtype= np.float32
        )

        #observation space: [normalized_price, inventory ratio, demand_signal,
        #                    competitor_avg, time_of_day, day_of_week, seasonality]

        self.observation_space= spaces.Box(
            low= np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
            high= np.array([2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
            dtype= np.float32
        )

        #for tracking performance
        self.episode_rewards= []
        self.price_history= []
        self.sales_history= []
        self.profit_history= []

    def reset(self, seed= None, options= None):
        """reset environment to initial state"""
        super().reset(seed=seed)

        # Generate unique episode ID
        self.episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        #reset state variables
        self.current_price=self.config["base_price"]
        self.inventory= self.config["initial_inventory"]
        self.time_step= 0
        self.total_revenue=0.0
        self.total_profit=0.0

        #reset competitor prices with some randomness
        base_competitor_price= self.config["base_price"]
        self.competitor_prices=[
            base_competitor_price+np.random.normal(0,5) for _ in range (3)
        ]

        #CLear History
        self.price_history= [self.current_price]
        self.sales_history=[]
        self.profit_history=[]

        # Log episode start to ELK Stack
        self.elk_logger.log_environment_state({
            "episode_id": self.episode_id,
            "event_subtype": "episode_start",
            "initial_price": self.current_price,
            "initial_inventory": self.inventory,
            "competitor_prices": self.competitor_prices,
            "config": self.config
        })

        observation=self._get_observation()
        info= self._get_info()

        return observation, info
    
    def step(self, action):
        """execute one time step within the environment"""
        #apply price change( action is a multiplier)
        price_multiplier= 1.0+ float(action[0])
        new_price= self.current_price*price_multiplier

        #ensure price stays within range(below cost or tooo high)
        new_price= np.clip(new_price,
                           self.config["base_cost"]*1.1,#atleast 10% markup
                           self.config["base_price"]*3.0)#max 3 times base price
        
        old_price = self.current_price
        self.current_price= new_price

        #SIMULATE MARKET RESPONSE
        demand= self._simulate_demand()
        sales= min(demand, self.inventory) #cant sell more than inventory

        #update inventory
        old_inventory = self.inventory
        self.inventory= max(0, self.inventory-sales)

        #calculate financials
        revenue= self.current_price*sales
        cost= self.config["base_cost"]* sales
        profit= revenue-cost

        self.total_revenue +=revenue
        self.total_profit += profit

        #CALCULATE REWARD
        reward= self._calculate_reward(profit, sales, demand)

        # Log action and state to ELK Stack
        self.elk_logger.log_action_taken({
            "episode_id": self.episode_id,
            "time_step": self.time_step,
            "action": float(action[0]),
            "price_multiplier": price_multiplier,
            "old_price": old_price,
            "new_price": self.current_price,
            "demand": demand,
            "sales": sales,
            "old_inventory": old_inventory,
            "new_inventory": self.inventory,
            "revenue": revenue,
            "profit": profit,
            "reward": reward,
            "competitor_prices": self.competitor_prices.copy()
        })

        #update competitors(they react to our pricing)
        self._update_competitors()

        #restock inventory( sSIMULATE SUPPLY CHAIN)
        self._restock_inventory()

        #update time
        self.time_step+=1

        #check if episode is done
        terminated= self. time_step >=self.config["episode_length"]
        truncated= self.inventory<=0 #out of stock

        #store history
        self.price_history.append(self.current_price)
        self.sales_history.append(sales)
        self.profit_history.append(profit)

        # Log environment state every few steps
        if self.time_step % 10 == 0 or terminated or truncated:
            self.elk_logger.log_environment_state({
                "episode_id": self.episode_id,
                "event_subtype": "periodic_state",
                "time_step": self.time_step,
                "current_price": self.current_price,
                "inventory": self.inventory,
                "total_revenue": self.total_revenue,
                "total_profit": self.total_profit,
                "competitor_prices": self.competitor_prices,
                "terminated": terminated,
                "truncated": truncated
            })

        # Log episode end
        if terminated or truncated:
            self._log_episode_end()

        #GET NEW OBSERVATION
        observation= self._get_observation()
        info= self._get_info()

        return observation, reward, terminated, truncated, info
    
    def _log_episode_end(self):
        """Log episode completion metrics to ELK Stack"""
        episode_summary = self.get_episode_summary()
        
        self.elk_logger.log_performance_metrics({
            "episode_id": self.episode_id,
            "event_subtype": "episode_complete",
            "total_steps": self.time_step,
            "final_inventory": self.inventory,
            **episode_summary,
            "price_history": self.price_history,
            "sales_history": self.sales_history,
            "profit_history": self.profit_history
        })
    
    def _simulate_demand(self)-> float:
        """simulate customer demand based on current market conditions"""
        #base demand with time of day effect
        time_factor= 0.8+0.4*np.sin(2*np.pi*(self.time_step%24)/24)
        base_demand= self.base_demand*time_factor

        #price elasticity effect (higher price= lower demand)
        price_ratio= self.current_price/ self.config["base_price"]
        price_effect= price_ratio**self.config["price_elasticity"]

        #COMPETITOR EFFECT(if commpetitors are cheaper our demand decreases)
        avg_competitor_price= np.mean(self.competitor_prices)
        competitor_ratio= self.current_price/ avg_competitor_price
        competitor_effect= competitor_ratio**-0.8 #NEGATIVE EFFECT OF BEING EXPENSIVE

        #seasonality (weekly and daily patterns)
        day_of_week= (self.time_step//24)%7
        weekly_seasonality= 1.0 + self.config["seasonality_amplitude"]*np.sin(2*np.pi* day_of_week/7)

        #random noise
        noise= 1.0 +np.random.normal(0, self.config["noise_level"])
        
        #COMBINING ALL EFFECTS
        demand= base_demand*price_effect*competitor_effect*weekly_seasonality*noise
        demand= max(0, demand) #demand cant be negative

        return demand
    
    def _calculate_reward(self, profit:float, sales:float, demand:float)-> float:
        """calculate reward for the agent"""
        #primary reward:profit
        profit_reward= profit/100.00 #can be scaled down for training stability

        #bonus for maintaining market share(selling close to demand)
        if demand> 0:
            market_share_bonus= (sales/ demand)*10.0
        else:
            market_share_bonus= 0.0

        #penalty for being too far from base price(avoid extreme prices)
        price_ratio= self.current_price/ self.config["base_price"]
        if price_ratio<0.7 or price_ratio>1.5:
            price_penalty= -20.0*abs(price_ratio-1.0)
        else:
            price_penalty= 0.0

        #inventory management bonus/penalty
        inventory_ratio= self.inventory/ self.config["max_inventory"]
        if inventory_ratio<0.1: #too low inventory
            inventory_penalty= -10.0
        elif inventory_ratio>0.9: #too high
            inventory_penalty= -5.0
        else:
            inventory_penalty=0.0

        total_reward= profit_reward+market_share_bonus+price_penalty+inventory_penalty

        return total_reward
    
    def _update_competitors(self):
        """update competitor prices as they react to our pricing"""
        for i, competitor_price in enumerate(self.competitor_prices):
            #SOME COMPETITORS COPY US OTHERS HAVE THEIR OWN STRATEGY
            if i==0: #copycat competitor
                target_price= self.current_price*0.95 #try to undercut us
                self.competitor_prices[i] += 0.1* (target_price-competitor_price)

            elif i==1: #premium competitor
                target_price= max(self.config["base_price"]*1.2, self.current_price*1.05)
                self.competitor_prices[i] += 0.05 * (target_price-competitor_price)

            else: #random competitor
                self.competitor_prices[i]+=np.random.normal(0,2)

            
            #keep prices reasonable
            self.competitor_prices[i]= np.clip(
                self.competitor_prices[i],
                self.config["base_cost"]*1.05,
                self.config["base_price"]*2.5)
    
    def _restock_inventory(self):
        """simulate inventory restocking"""
        #simple restocking: add inventory each time step
        restock_amount= min(20, self.config["max_inventory"]- self.inventory)
        self.inventory += restock_amount
    
    def _get_observation(self)-> np.ndarray:
        """get current observation state"""

        #normalise values to [0,1] range for better training
        normalised_price =self.current_price/ self.config["base_price"]
        inventory_ratio= self.inventory/ self.config["max_inventory"]
        demand_signal =self._simulate_demand()/ self.config["max_demand"]
        competitor_avg= np.mean(self.competitor_prices)/ self.config["base_price"]

        #time features
        hour_of_day= (self.time_step%24)/24.0
        day_of_week= ((self.time_step//24)%7)/7.0

        #seasonality
        seasonality= 0.5+0.5 *np.sin(2*np.pi * (self.time_step%168)/168) #weekly cycle

        observation= np.array([
            normalised_price,
            inventory_ratio,
            demand_signal,
            competitor_avg,
            hour_of_day,
            day_of_week,
            seasonality
            ], dtype= np.float32)
        
        return observation
    
    def _get_info(self)->Dict:
        """get additional information about the current state"""
        return{
            "episode_id": self.episode_id,
            "current_price": self.current_price,
            "inventory": self.inventory,
            "total_revenue": self.total_revenue,
            "total_profit": self.total_profit,
            "competitor_prices": self.competitor_prices,
            "time_step": self.time_step
        }
    
    def render(self, mode="human"):
        """render the environment (optional visualisation)"""
        if len(self.price_history) >1:
            plt.figure(figsize=(12,8))

            #plot price history
            plt.subplot(2,2,1)
            plt.plot(self.price_history, label="Our Price")
            plt.axhline(y=np.mean(self.competitor_prices), color="r", linestyle= "--", label= "Competitor Avg")
            plt.title("Price History")
            plt.legend()

            #plot sales history
            plt.subplot(2,2,2)
            plt.plot(self.sales_history, label= "Sales History")
            plt.title("Sales History")
            plt.legend()

            #plot profit history
            plt.subplot(2,2,3)
            plt.plot(self.profit_history, label= "Profit")
            plt.title("Profit History")
            plt.legend()

            #plot cumulative profit
            plt.subplot(2,2,4)
            cumulative_profit= np.cumsum(self.profit_history)
            plt.plot(cumulative_profit, label= "Cumulative Profit")
            plt.title("Cumulative Profit")
            plt.legend()

            plt.tight_layout()
            plt.show()

    def get_episode_summary(self)-> Dict:
        """get summary stats for completed episodes"""
        if not self.price_history:
            return{}
        
        return{
            "total_profit": sum(self.profit_history),
            "total_revenue":self.total_revenue,
            "avg_price": np.mean(self.price_history),
            "price_volatility":np.std(self.price_history),
            "total_sales":sum(self.sales_history),
            "avg_profit_per_sale": sum(self.profit_history)/ max(sum(self.sales_history), 1),
            "episode_length": len(self.profit_history)
        }

#EXAMPLE USAGE AND TESTING
if __name__=="__main__":
    #create environment
    env= DynamicPricingEnv()

    #testing
    obs, info= env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    #take a few random actions
    for i in range (10):
        action= env.action_space.sample()  #random action
        obs, reward, terminated, truncated, info= env.step(action)

        print(f"step{i+1}:")
        print(f"Action: {action[0]:.3f} (price change)")
        print(f" new price: ${info['current_price']:.2f}")
        print(f" Reward: {reward:.2f}")
        print(f"Inventory: {info['inventory']}")
        print(f" Profit: ${info['total_profit']: .2f}")

        if terminated or truncated:
            print("episode ended!!")
            break

    #show episode summary
    summary= env.get_episode_summary()
    print("Episode Summary:", summary)

    #render the environment(optional)
    env.render()