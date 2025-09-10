"""
Kafka-Enabled Pricing Environment
Integrates with the pricing environment to consume pricing actions and publish market updates
"""
import asyncio
import threading
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque
import json

from pricing_environment import DynamicPricingEnv
from kafka_config import (
    KafkaConfig, KafkaProducerWrapper, KafkaConsumerWrapper,
    create_market_event_message, create_system_metrics_message,
    KafkaHealthChecker, KafkaTopicManager
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KafkaPricingEnvironment:
    """Pricing Environment that consumes pricing actions and produces market events via Kafka"""
    
    def __init__(self, env_config: Dict = None, kafka_config: Dict = None):
        self.env_config = env_config or {
            'base_price': 100.0,
            'base_cost': 70.0,
            'max_inventory': 1000,
            'initial_inventory': 500,
            'price_elasticity': -1.5,
            'competitor_reactivity': 0.3,
            'episode_length': 1000  # Long running environment
        }
        
        self.kafka_config = kafka_config or {
            'environment_id': 'kafka_pricing_env_001',
            'update_interval': 3.0,      # Seconds between market updates
            'price_update_timeout': 10.0, # Max wait for price updates
            'batch_size': 1,             # Market events per batch
            'auto_advance': True         # Auto-advance time if no pricing actions
        }
        
        # Initialize pricing environment
        self.env = DynamicPricingEnv(self.env_config)
        self.current_obs, self.current_info = self.env.reset()
        
        # Kafka components
        self.producer = KafkaProducerWrapper()
        self.consumer = None
        
        # State management
        self.running = False
        self.current_price = self.env_config['base_price']
        self.pending_pricing_actions = deque(maxlen=10)
        self.market_updates_sent = []
        self.last_update_time = time.time()
        
        # Performance tracking
        self.stats = {
            'total_pricing_actions': 0,
            'total_market_updates': 0,
            'avg_processing_time': 0.0,
            'environment_steps': 0,
            'total_reward': 0.0,
            'total_profit': 0.0,
            'price_changes': deque(maxlen=100),
            'demand_history': deque(maxlen=100),
            'sales_history': deque(maxlen=100),
            'last_market_update': None
        }
        
        # Threading
        self.consumer_thread = None
        self.environment_thread = None
        
    def setup_kafka_consumer(self):
        """Set up Kafka consumer for pricing actions"""
        def pricing_action_callback(topic: str, message: Dict[str, Any]):
            if topic == KafkaConfig.PRICING_ACTIONS_TOPIC:
                self._handle_pricing_action(message)
        
        self.consumer = KafkaConsumerWrapper(
            group_id=self.kafka_config['environment_id'],
            topics=[KafkaConfig.PRICING_ACTIONS_TOPIC]
        )
        
        if not self.consumer.subscribe():
            raise Exception("Failed to subscribe to pricing actions topic")
        
        logger.info("Kafka consumer for pricing actions set up successfully")
    
    def _handle_pricing_action(self, message: Dict[str, Any]):
        """Handle incoming pricing action from RL agent"""
        try:
            action_type = message.get('action_type')
            if action_type != 'price_update':
                return
            
            new_price = message.get('price')
            agent_info = message.get('agent_info', {})
            
            if new_price and isinstance(new_price, (int, float)):
                pricing_action = {
                    'timestamp': message.get('timestamp'),
                    'price': float(new_price),
                    'agent_id': agent_info.get('agent_id', 'unknown'),
                    'confidence': agent_info.get('confidence', 0.5),
                    'decision_type': agent_info.get('decision_type', 'unknown'),
                    'market_context': agent_info.get('market_context', {})
                }
                
                self.pending_pricing_actions.append(pricing_action)
                self.stats['total_pricing_actions'] += 1
                
                logger.info(f"Received pricing action: ${new_price:.2f} from agent {agent_info.get('agent_id', 'unknown')} "
                          f"(confidence: {agent_info.get('confidence', 0.5):.2f})")
            else:
                logger.warning(f"Invalid price in pricing action: {new_price}")
        
        except Exception as e:
            logger.error(f"Error handling pricing action: {e}")
    
    def _get_next_price(self) -> float:
        """Get next price from pending actions or use current price"""
        if self.pending_pricing_actions:
            action = self.pending_pricing_actions.popleft()
            new_price = action['price']
            
            # Apply price constraints
            min_price = self.env_config['base_cost'] * 1.1  # At least 10% markup
            max_price = self.env_config['base_price'] * 3.0  # Max 3x base price
            new_price = np.clip(new_price, min_price, max_price)
            
            # Track price change
            price_change = (new_price - self.current_price) / self.current_price
            self.stats['price_changes'].append(price_change)
            
            return new_price
        
        # No pending actions - use current price or auto-advance
        if self.kafka_config['auto_advance']:
            # Small random variation to keep market moving
            variation = np.random.normal(0, 0.01)  # 1% standard deviation
            new_price = self.current_price * (1 + variation)
            return np.clip(new_price, 
                          self.env_config['base_cost'] * 1.1,
                          self.env_config['base_price'] * 2.0)
        
        return self.current_price
    
    def _step_environment(self, price: float) -> Dict[str, Any]:
        """Step the environment with given price and return market data"""
        start_time = time.time()
        
        try:
            # Convert price to action for environment
            # Environment expects action as price multiplier
            current_env_price = getattr(self.env, 'current_price', self.env_config['base_price'])
            price_multiplier = (price / current_env_price) - 1.0
            price_multiplier = np.clip(price_multiplier, -0.2, 0.2)  # Limit to ±20%
            
            action = np.array([price_multiplier], dtype=np.float32)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Update state
            self.current_obs = obs
            self.current_info = info
            self.current_price = info.get('current_price', price)
            
            # Update statistics
            self.stats['environment_steps'] += 1
            self.stats['total_reward'] += reward
            self.stats['total_profit'] = info.get('total_profit', 0)
            
            # Track demand and sales
            if hasattr(self.env, 'sales_history') and self.env.sales_history:
                latest_sales = self.env.sales_history[-1]
                self.stats['sales_history'].append(latest_sales)
            
            if hasattr(self.env, 'price_history') and len(self.env.price_history) > 1:
                # Estimate demand from environment (if available)
                demand_estimate = info.get('demand', 50.0)
                self.stats['demand_history'].append(demand_estimate)
            
            # Create market data
            market_data = {
                'our_price': float(self.current_price),
                'competitor_prices': getattr(self.env, 'competitor_prices', [95.0, 105.0, 98.0]),
                'demand': float(self.stats['demand_history'][-1] if self.stats['demand_history'] else 50.0),
                'sales': float(self.stats['sales_history'][-1] if self.stats['sales_history'] else 0.0),
                'inventory': float(info.get('inventory', 500)),
                'reward': float(reward),
                'total_profit': float(self.stats['total_profit']),
                'time_step': self.stats['environment_steps'],
                'hour_of_day': self.stats['environment_steps'] % 24,
                'day_of_week': (self.stats['environment_steps'] // 24) % 7,
                'external_events': [],  # Could be enhanced
                'demand_multiplier': 1.0  # Could be enhanced
            }
            
            # Reset environment if episode ended
            if terminated or truncated:
                logger.info("Environment episode ended, resetting...")
                self.current_obs, self.current_info = self.env.reset()
            
            # Update processing time
            processing_time = time.time() - start_time
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['environment_steps'] - 1) + processing_time) /
                self.stats['environment_steps']
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error stepping environment: {e}")
            # Return safe fallback market data
            return {
                'our_price': float(self.current_price),
                'competitor_prices': [95.0, 105.0, 98.0],
                'demand': 50.0,
                'sales': 0.0,
                'inventory': 500.0,
                'reward': 0.0,
                'total_profit': 0.0,
                'time_step': self.stats['environment_steps'],
                'hour_of_day': 0,
                'day_of_week': 0,
                'external_events': [],
                'demand_multiplier': 1.0,
                'error': str(e)
            }
    
    def _publish_market_update(self, market_data: Dict[str, Any]) -> bool:
        """Publish market update to Kafka"""
        try:
            # Create standardized message
            message = create_market_event_message(market_data)
            
            # Add environment metadata
            message['producer_id'] = self.kafka_config['environment_id']
            message['sequence_number'] = self.stats['total_market_updates'] + 1
            message['environment_step'] = self.stats['environment_steps']
            
            # Publish to Kafka
            success = self.producer.produce_message(
                topic=KafkaConfig.MARKET_EVENTS_TOPIC,
                message=message,
                key=f"market_env_{int(time.time())}"
            )
            
            if success:
                self.market_updates_sent.append(market_data)
                self.stats['total_market_updates'] += 1
                self.stats['last_market_update'] = market_data
                self.last_update_time = time.time()
                
                logger.debug(f"Published market update #{self.stats['total_market_updates']}: "
                           f"Price=${market_data['our_price']:.2f}, "
                           f"Demand={market_data['demand']:.1f}")
            else:
                logger.error("Failed to publish market update")
            
            return success
            
        except Exception as e:
            logger.error(f"Error publishing market update: {e}")
            return False
    
    def _publish_environment_metrics(self):
        """Publish environment performance metrics"""
        try:
            metrics = {
                'environment_id': self.kafka_config['environment_id'],
                'total_pricing_actions': self.stats['total_pricing_actions'],
                'total_market_updates': self.stats['total_market_updates'],
                'environment_steps': self.stats['environment_steps'],
                'avg_processing_time_ms': self.stats['avg_processing_time'] * 1000,
                'total_reward': self.stats['total_reward'],
                'total_profit': self.stats['total_profit'],
                'current_price': self.current_price,
                'avg_demand': float(np.mean(self.stats['demand_history'])) if self.stats['demand_history'] else 0.0,
                'avg_sales': float(np.mean(self.stats['sales_history'])) if self.stats['sales_history'] else 0.0,
                'price_volatility': float(np.std(self.stats['price_changes'])) if self.stats['price_changes'] else 0.0,
                'pending_actions': len(self.pending_pricing_actions)
            }
            
            message = create_system_metrics_message(metrics)
            message['metrics_type'] = 'environment_performance'
            
            self.producer.produce_message(
                topic=KafkaConfig.SYSTEM_METRICS_TOPIC,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error publishing environment metrics: {e}")
    
    def _consumer_loop(self):
        """Consumer loop - processes pricing actions"""
        logger.info("Environment consumer loop started")
        
        self.consumer.start_consuming()
        
        while self.running:
            try:
                self.consumer.consume_messages(
                    callback=lambda topic, msg: None,  # Callback handled in setup
                    timeout=1.0,
                    max_messages=10
                )
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                time.sleep(1)
        
        logger.info("Environment consumer loop stopped")
    
    def _environment_loop(self):
        """Environment loop - steps environment and publishes market updates"""
        logger.info("Environment update loop started")
        
        while self.running:
            try:
                current_time = time.time()
                time_since_update = current_time - self.last_update_time
                
                # Check if it's time for market update
                if time_since_update >= self.kafka_config['update_interval']:
                    
                    # Get next price
                    next_price = self._get_next_price()
                    
                    # Step environment
                    market_data = self._step_environment(next_price)
                    
                    # Publish market update
                    self._publish_market_update(market_data)
                    
                    # Publish metrics every 10 updates
                    if self.stats['total_market_updates'] % 10 == 0:
                        self._publish_environment_metrics()
                    
                    # Log progress
                    if self.stats['total_market_updates'] % 20 == 0:
                        logger.info(f"Published {self.stats['total_market_updates']} market updates, "
                                  f"Environment steps: {self.stats['environment_steps']}")
                
                time.sleep(0.5)  # Check twice per second
                
            except Exception as e:
                logger.error(f"Error in environment loop: {e}")
                time.sleep(2)
        
        logger.info("Environment update loop stopped")
    
    def start_environment(self):
        """Start the Kafka pricing environment"""
        if self.running:
            logger.warning("Environment already running")
            return
        
        logger.info("Starting Kafka Pricing Environment...")
        
        # Setup Kafka consumer
        self.setup_kafka_consumer()
        
        # Start environment
        self.running = True
        self.last_update_time = time.time()
        
        # Start consumer thread
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()
        
        # Start environment thread
        self.environment_thread = threading.Thread(target=self._environment_loop, daemon=True)
        self.environment_thread.start()
        
        logger.info("Kafka Pricing Environment started successfully")
        
        # Publish initial market state
        initial_market_data = {
            'our_price': float(self.current_price),
            'competitor_prices': getattr(self.env, 'competitor_prices', [95.0, 105.0, 98.0]),
            'demand': 50.0,
            'sales': 0.0,
            'inventory': float(self.env_config['initial_inventory']),
            'reward': 0.0,
            'total_profit': 0.0,
            'time_step': 0,
            'hour_of_day': 0,
            'day_of_week': 0,
            'external_events': [],
            'demand_multiplier': 1.0
        }
        self._publish_market_update(initial_market_data)
    
    def stop_environment(self):
        """Stop the pricing environment"""
        if not self.running:
            logger.warning("Environment not running")
            return
        
        logger.info("Stopping Kafka Pricing Environment...")
        
        self.running = False
        
        # Wait for threads to complete
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5.0)
        
        if self.environment_thread and self.environment_thread.is_alive():
            self.environment_thread.join(timeout=5.0)
        
        # Close Kafka components
        if self.consumer:
            self.consumer.close()
        
        self.producer.flush(timeout=5.0)
        self.producer.close()
        
        logger.info("Kafka Pricing Environment stopped")
        
        return self.get_environment_stats()
    
    def get_environment_stats(self) -> Dict[str, Any]:
        """Get current environment statistics"""
        return {
            **self.stats,
            'current_price': self.current_price,
            'pending_actions_count': len(self.pending_pricing_actions),
            'market_updates_sent_count': len(self.market_updates_sent),
            'environment_running': self.running,
            'last_update_time': self.last_update_time,
            'avg_reward_per_step': self.stats['total_reward'] / max(1, self.stats['environment_steps'])
        }


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Pricing Environment')
    parser.add_argument('--duration', type=int, default=120,
                      help='How long to run the environment (seconds)')
    parser.add_argument('--update-interval', type=float, default=4.0,
                      help='Interval between market updates (seconds)')
    parser.add_argument('--auto-advance', action='store_true', default=True,
                      help='Auto-advance time if no pricing actions received')
    
    args = parser.parse_args()
    
    # Check Kafka connection
    if not KafkaHealthChecker.wait_for_kafka(max_retries=10):
        print("Failed to connect to Kafka. Make sure it's running on localhost:9092")
        exit(1)
    
    # Create topics
    topic_manager = KafkaTopicManager()
    if not topic_manager.create_topics_if_not_exist():
        print("Failed to create Kafka topics")
        exit(1)
    
    # Configure environment
    env_config = {
        'base_price': 120.0,
        'base_cost': 80.0,
        'price_elasticity': -1.8,
        'competitor_reactivity': 0.4,
        'episode_length': 2000  # Long running
    }
    
    kafka_config = {
        'environment_id': 'pricing_env_demo',
        'update_interval': args.update_interval,
        'auto_advance': args.auto_advance
    }
    
    # Create and start environment
    environment = KafkaPricingEnvironment(env_config=env_config, kafka_config=kafka_config)
    
    try:
        environment.start_environment()
        
        print(f"Pricing Environment running for {args.duration} seconds...")
        print("Publishing market updates and waiting for pricing actions...")
        
        # Run for specified duration
        time.sleep(args.duration)
        
    except KeyboardInterrupt:
        print("\nStopping due to keyboard interrupt...")
    
    finally:
        # Stop environment and show final stats
        final_stats = environment.stop_environment()
        print(f"\nEnvironment Performance Summary:")
        print(f"Total Market Updates: {final_stats.get('total_market_updates', 0)}")
        print(f"Total Pricing Actions: {final_stats.get('total_pricing_actions', 0)}")
        print(f"Environment Steps: {final_stats.get('environment_steps', 0)}")
        print(f"Total Profit: ${final_stats.get('total_profit', 0):.2f}")
        print(f"Avg Processing Time: {final_stats.get('avg_processing_time', 0)*1000:.2f}ms")
        if final_stats.get('price_changes'):
            print(f"Price Volatility: {np.std(list(final_stats['price_changes']))*100:.2f}%")
    
    print("Done!")