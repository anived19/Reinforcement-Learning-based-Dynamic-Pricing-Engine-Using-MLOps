"""
Kafka-Enabled RL Agent for Dynamic Pricing
Integrates with the RL training pipeline to consume market events and produce pricing actions
"""
import asyncio
import threading
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Deque
from collections import deque
import joblib
import os
import torch

from performance_logger import PerformanceLogger
from stable_baselines3 import PPO, A2C
from pricing_environment import DynamicPricingEnv
from kafka_config import (
    KafkaConfig, KafkaProducerWrapper, KafkaConsumerWrapper,
    create_pricing_action_message, create_system_metrics_message,
    KafkaHealthChecker, KafkaTopicManager
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KafkaRLAgent:
    """RL Agent that consumes market events and produces pricing actions via Kafka"""
    
    def __init__(self, model_path: str = None, agent_config: Dict = None):
        self.agent_config = agent_config or {
            'agent_id': 'rl_pricing_agent_001',
            'decision_interval': 5.0,  # Seconds between pricing decisions
            'confidence_threshold': 0.3,  # Minimum confidence for price changes
            'max_price_change': 0.15,  # Maximum price change per step (15%)
            'observation_window': 10,   # Number of recent observations to keep
            'warm_up_steps': 5,        # Steps before making decisions
            'fallback_pricing': True   # Use fallback pricing if model fails
        }

        self.performance_logger = PerformanceLogger()
        
        # Model management
        self.model = None
        self.model_path = model_path
        self.env = None
        self.current_observation = None
        self.observation_history = deque(maxlen=self.agent_config['observation_window'])
        
        # Kafka components
        self.producer = KafkaProducerWrapper()
        self.consumer = None
        
        # State management
        self.running = False
        self.market_events_received = []
        self.pricing_actions_sent = []
        self.current_price = 100.0  # Default price
        self.base_price = 100.0
        
        # Performance tracking
        self.stats = {
            'total_market_events': 0,
            'total_pricing_actions': 0,
            'avg_decision_time': 0.0,
            'model_predictions': 0,
            'fallback_decisions': 0,
            'confidence_scores': deque(maxlen=100),
            'price_changes': deque(maxlen=100),
            'last_decision_time': None
        }
        
        # Threading
        self.consumer_thread = None
        self.decision_thread = None
        
    def load_model(self, model_path: str = None):
        """Load trained RL model"""
        model_path = model_path or self.model_path
        
        if not model_path or not os.path.exists(f"{model_path}.zip"):
            logger.warning(f"Model file not found: {model_path}.zip")
            logger.info("Running without trained model - using fallback pricing strategy")
            return False
        
        try:
            # Try to load PPO first, then A2C
            try:
                self.model = PPO.load(model_path)
                logger.info(f"Loaded PPO model from {model_path}")
            except:
                self.model = A2C.load(model_path)
                logger.info(f"Loaded A2C model from {model_path}")
            
            # Create matching environment for observation preprocessing
            self.env = DynamicPricingEnv()
            self.env.reset()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            return False
    
    def setup_kafka_consumer(self):
        """Set up Kafka consumer for market events"""
        def market_event_callback(topic: str, message: Dict[str, Any]):
            if topic == KafkaConfig.MARKET_EVENTS_TOPIC:
                self._handle_market_event(message)
        
        self.consumer = KafkaConsumerWrapper(
            group_id=self.agent_config['agent_id'],
            topics=[KafkaConfig.MARKET_EVENTS_TOPIC]
        )
        
        if not self.consumer.subscribe():
            raise Exception("Failed to subscribe to market events topic")
        
        logger.info("Kafka consumer for market events set up successfully")
    
    def _handle_market_event(self, message: Dict[str, Any]):
        """Handle incoming market event"""
        try:
            event_type = message.get('event_type')
            if event_type != 'market_update':
                return
            
            market_data = message.get('data', {})
            
            # Extract relevant information
            market_info = {
                'timestamp': message.get('timestamp'),
                'our_price': market_data.get('our_price', self.current_price),
                'competitor_prices': market_data.get('competitor_prices', []),
                'demand': market_data.get('demand', 0),
                'sales': market_data.get('sales', 0),
                'inventory': market_data.get('inventory', 0),
                'external_events': market_data.get('external_events', []),
                'demand_multiplier': market_data.get('demand_multiplier', 1.0),
                'hour_of_day': market_data.get('hour_of_day', 0),
                'day_of_week': market_data.get('day_of_week', 0)
            }

            self.performance_logger.log_market_event(market_info)
            
            # Update current price
            if 'our_price' in market_data:
                self.current_price = market_data['our_price']
            
            # Store event
            self.market_events_received.append(market_info)
            self.observation_history.append(market_info)
            
            self.stats['total_market_events'] += 1
            
            logger.debug(f"Received market event: Price=${market_info['our_price']:.2f}, "
                        f"Demand={market_info['demand']:.1f}, Sales={market_info['sales']:.1f}")
            
        except Exception as e:
            logger.error(f"Error handling market event: {e}")

    
    def _convert_to_observation(self, market_info: Dict[str, Any]) -> np.ndarray:
        """Convert market information to RL environment observation"""
        try:
            if not self.env:
                # Fallback observation creation
                return np.array([
                    market_info.get('our_price', 100.0) / 100.0,  # normalized price
                    market_info.get('inventory', 500) / 1000.0,   # inventory ratio
                    market_info.get('demand', 50) / 100.0,        # demand signal
                    np.mean(market_info.get('competitor_prices', [100.0])) / 100.0,  # competitor avg
                    market_info.get('hour_of_day', 0) / 24.0,     # time features
                    market_info.get('day_of_week', 0) / 7.0,
                    0.5  # seasonality placeholder
                ], dtype=np.float32)
            
            # Use environment's observation format
            base_price = getattr(self.env, 'config', {}).get('base_price', 100.0)
            max_inventory = getattr(self.env, 'config', {}).get('max_inventory', 1000)
            max_demand = getattr(self.env, 'config', {}).get('max_demand', 100)
            
            normalized_price = market_info.get('our_price', base_price) / base_price
            inventory_ratio = market_info.get('inventory', max_inventory/2) / max_inventory
            demand_signal = market_info.get('demand', 50) / max_demand
            
            competitor_prices = market_info.get('competitor_prices', [base_price])
            competitor_avg = np.mean(competitor_prices) / base_price if competitor_prices else 1.0
            
            hour_of_day = market_info.get('hour_of_day', 0) / 24.0
            day_of_week = market_info.get('day_of_week', 0) / 7.0
            
            # Simple seasonality based on time
            seasonality = 0.5 + 0.5 * np.sin(2 * np.pi * hour_of_day)
            
            observation = np.array([
                normalized_price,
                inventory_ratio,
                demand_signal,
                competitor_avg,
                hour_of_day,
                day_of_week,
                seasonality
            ], dtype=np.float32)
            
            # Clip to valid ranges
            observation = np.clip(observation, 0.0, 2.0)
            
            return observation
            
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            # Return safe default observation
            return np.array([1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.5], dtype=np.float32)
    
    def _make_pricing_decision(self, market_info: Dict[str, Any]) -> Dict[str, Any]:
        """Make pricing decision based on market information"""
        start_time = time.time()
    
        try:
            if self.model and len(self.observation_history) >= self.agent_config['warm_up_steps']:
            # Use trained RL model
                observation = self._convert_to_observation(market_info)
                action, _ = self.model.predict(observation, deterministic=False)

                # Convert observation to tensor
                obs_tensor = torch.tensor(observation, dtype=torch.float32)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)

                # Forward pass through the policy to get action distribution
                with torch.no_grad():
                    distribution = self.model.policy.get_distribution(obs_tensor)
                    dist_obj = distribution.distribution

                    if hasattr(dist_obj, "probs"):  # Discrete (Categorical)
                        action_probs = dist_obj.probs.cpu().numpy()[0]
                        confidence = float(np.max(action_probs))
                    else:  # Continuous (Normal)
                        mean = dist_obj.mean.cpu().numpy()[0]
                        std = dist_obj.stddev.cpu().numpy()[0]
                        # Define "confidence" as inverse of std (uncertainty)
                        confidence = float(1.0 / (1.0 + np.mean(std)))
            
                # Apply constraints
                max_change = self.agent_config['max_price_change']
                raw_action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
                # Scale action with tanh → keeps values in [-max_change, +max_change]
                scaled_action=raw_action * max_change * 2.0
                price_multiplier = 1.0 + scaled_action
            
                new_price = self.current_price * price_multiplier
            
                # Ensure price is reasonable
                min_price = self.base_price * 0.7
                max_price = self.base_price * 2.0
                new_price = np.clip(new_price, min_price, max_price)
            
                decision_type = 'rl_model'
                self.stats['model_predictions'] += 1

                logger.info(f"Model prediction - Raw action: {raw_action:.3f}, "
                             f"Scaled action: {scaled_action:.3f}, "
                             f"Confidence: {confidence:.3f}, "
                             f"STD: {np.mean(std):.3f}, "
                             f"Price change: {((new_price - self.current_price) / self.current_price)*100:.1f}%")
            
            else:
                # Fallback pricing strategy
                new_price = self._fallback_pricing_strategy(market_info)
                confidence = 0.5
                decision_type = 'fallback'
                self.stats['fallback_decisions'] += 1
        
            # Calculate decision time
            decision_time = time.time() - start_time
            self.stats['avg_decision_time'] = (
                (self.stats['avg_decision_time'] * self.stats['total_pricing_actions'] + decision_time) /
                (self.stats['total_pricing_actions'] + 1)
            )
        
            # Store performance metrics
            price_change_percent = (new_price - self.current_price) / self.current_price
            self.stats['confidence_scores'].append(confidence)
            self.stats['price_changes'].append(price_change_percent)
        
            decision = {
                'price': float(new_price),
                'confidence': float(confidence),
                'decision_type': decision_type,
                'price_change_percent': float(price_change_percent),
                'decision_time_ms': float(decision_time * 1000),
                'market_context': {
                    'previous_price': float(self.current_price),
                    'demand': market_info.get('demand', 0),
                    'inventory': market_info.get('inventory', 0),
                    'competitor_avg': float(np.mean(market_info.get('competitor_prices', [self.current_price])))
                }
            }
        
            return decision
        
        except Exception as e:
            logger.error(f"Error making pricing decision: {e}")
            # Emergency fallback
            return {
                'price': self.current_price,
                'confidence': 0.1,
                'decision_type': 'emergency_fallback',
                'price_change_percent': 0.0,
                'error': str(e)
            }

    
    def _fallback_pricing_strategy(self, market_info: Dict[str, Any]) -> float:
        """Fallback pricing strategy when RL model is not available"""
        try:
            current_price = market_info.get('our_price', self.current_price)
            demand = market_info.get('demand', 50)
            inventory = market_info.get('inventory', 500)
            competitor_prices = market_info.get('competitor_prices', [current_price])
            
            # Simple rule-based pricing
            competitor_avg = np.mean(competitor_prices)
            
            # Base adjustment on demand and competition
            if demand > 80:  # High demand
                if current_price < competitor_avg * 0.95:
                    return current_price * 1.05  # Increase price
            elif demand < 30:  # Low demand
                if current_price > competitor_avg * 1.05:
                    return current_price * 0.98  # Decrease price
            
            # Inventory-based adjustments
            if inventory < 100:  # Low inventory
                return current_price * 1.03
            elif inventory > 800:  # High inventory
                return current_price * 0.99
            
            # Default: small random adjustment
            return current_price * (1.0 + np.random.normal(0, 0.01))
            
        except Exception as e:
            logger.error(f"Error in fallback pricing: {e}")
            return self.current_price
    
    def _publish_pricing_action(self, decision: Dict[str, Any]) -> bool:
        """Publish pricing action to Kafka"""
        try:
            # Create pricing action message
            message = create_pricing_action_message(
                price=decision['price'],
                agent_info={
                    'agent_id': self.agent_config['agent_id'],
                    'confidence': decision['confidence'],
                    'decision_type': decision['decision_type'],
                    'price_change_percent': decision['price_change_percent'],
                    'decision_time_ms': decision.get('decision_time_ms', 0),
                    'market_context': decision.get('market_context', {})
                }
            )
            
            # Publish to Kafka
            success = self.producer.produce_message(
                topic=KafkaConfig.PRICING_ACTIONS_TOPIC,
                message=message,
                key=f"pricing_{int(time.time())}"
            )
            
            if success:
                self.performance_logger.log_pricing_action(decision)
                self.pricing_actions_sent.append(decision)
                self.stats['total_pricing_actions'] += 1
                self.stats['last_decision_time'] = datetime.utcnow().isoformat()
                
                logger.info(f"Published pricing action: ${decision['price']:.2f} "
                          f"(confidence: {decision['confidence']:.2f}, "
                          f"change: {decision['price_change_percent']*100:.1f}%)")
            else:
                logger.error("Failed to publish pricing action")
            
            return success
            
        except Exception as e:
            logger.error(f"Error publishing pricing action: {e}")
            return False
    
    def _publish_agent_metrics(self):
        """Publish agent performance metrics"""
        try:
            metrics = {
                'agent_id': self.agent_config['agent_id'],
                'total_market_events': self.stats['total_market_events'],
                'total_pricing_actions': self.stats['total_pricing_actions'],
                'avg_decision_time_ms': self.stats['avg_decision_time'] * 1000,
                'model_predictions': self.stats['model_predictions'],
                'fallback_decisions': self.stats['fallback_decisions'],
                'avg_confidence': float(np.mean(self.stats['confidence_scores'])) if self.stats['confidence_scores'] else 0.0,
                'avg_price_change_percent': float(np.mean(self.stats['price_changes'])) if self.stats['price_changes'] else 0.0,
                'current_price': self.current_price,
                'model_loaded': self.model is not None,
                'observation_history_size': len(self.observation_history)
            }
            
            message = create_system_metrics_message(metrics)
            message['metrics_type'] = 'agent_performance'
            
            self.producer.produce_message(
                topic=KafkaConfig.SYSTEM_METRICS_TOPIC,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error publishing agent metrics: {e}")
    
    def _consumer_loop(self):
        """Consumer loop - processes market events"""
        logger.info("RL Agent consumer loop started")
    
        try:
            self.consumer.start_consuming()
        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            return
    
        while self.running:
            try:
                self.consumer.consume_messages(
                    callback=self._process_message,  # Use explicit callback
                    timeout=1.0,
                    max_messages=20
            )
                time.sleep(0.1)  # Small delay to prevent CPU spinning
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                time.sleep(1)
    
        logger.info("RL Agent consumer loop stopped")

    def _process_message(self, topic, message):
        """Process incoming message"""
        try:
            logger.debug(f"RL Agent received message from topic {topic}")
            if topic == 'market_events':
                self._handle_market_event(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _decision_loop(self):
        """Decision loop - makes pricing decisions periodically"""
        logger.info("RL Agent decision loop started")
        
        while self.running:
            try:
                if (len(self.observation_history) > 0 and 
                    (time.time() - getattr(self, '_last_decision_time', 0)) >= 
                    self.agent_config['decision_interval']):
                    
                    # Get latest market information
                    latest_market_info = self.observation_history[-1]
                    
                    # Make pricing decision
                    decision = self._make_pricing_decision(latest_market_info)
                    
                    # Publish decision if confidence is high enough
                    if decision['confidence'] >= self.agent_config['confidence_threshold']:
                        self._publish_pricing_action(decision)
                        self._last_decision_time = time.time()
                    else:
                        logger.debug(f"Skipping decision due to low confidence: {decision['confidence']:.2f}")
                
                # Publish metrics every 10 decisions
                if self.stats['total_pricing_actions'] % 10 == 0 and self.stats['total_pricing_actions'] > 0:
                    self._publish_agent_metrics()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in decision loop: {e}")
                time.sleep(2)
        
        logger.info("RL Agent decision loop stopped")
    
    def start_agent(self):
        """Start the RL agent"""
        if self.running:
            logger.warning("RL Agent already running")
            return
        
        logger.info("Starting Kafka RL Agent...")
        
        # Load model
        if self.model_path:
            self.load_model()
        
        # Setup Kafka consumer
        self.setup_kafka_consumer()
        
        # Start agent
        self.running = True
        self._last_decision_time = 0
        
        # Start consumer thread
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()
        
        # Start decision thread
        self.decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self.decision_thread.start()
        
        logger.info("RL Agent started successfully")
    
    def stop_agent(self):
        """Stop the RL agent"""
        if not self.running:
            logger.warning("RL Agent not running")
            return
        
        logger.info("Stopping RL Agent...")
        
        self.running = False
        
        # Wait for threads to complete
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5.0)
        
        if self.decision_thread and self.decision_thread.is_alive():
            self.decision_thread.join(timeout=5.0)
        
        # Close Kafka components
        if self.consumer:
            self.consumer.close()
        
        self.producer.flush(timeout=5.0)
        self.producer.close()
        
        logger.info("RL Agent stopped")

        self.performance_logger.save_final_summary()
        self.performance_logger.create_performance_graphs()
        self.performance_logger.create_advanced_analysis()
        report_file = self.performance_logger.generate_performance_report()
        
        return self.get_agent_stats()
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get current agent statistics"""
        return {
            **self.stats,
            'current_price': self.current_price,
            'model_loaded': self.model is not None,
            'observation_history_size': len(self.observation_history),
            'market_events_buffer_size': len(self.market_events_received),
            'pricing_actions_buffer_size': len(self.pricing_actions_sent),
            'agent_running': self.running
        }


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka RL Agent')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to trained RL model (without .zip extension)')
    parser.add_argument('--duration', type=int, default=120,
                      help='How long to run the agent (seconds)')
    parser.add_argument('--decision-interval', type=float, default=8.0,
                      help='Interval between pricing decisions (seconds)')
    
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
    
    # Configure agent
    agent_config = {
        'agent_id': 'rl_agent_demo',
        'decision_interval': args.decision_interval,
        'confidence_threshold': 0.3,
        'max_price_change': 0.1,
        'observation_window':50,
        'fallback_pricing': True
    }
    
    # Create and start agent
    agent = KafkaRLAgent(model_path=args.model_path, agent_config=agent_config)
    
    try:
        agent.start_agent()
        
        print(f"RL Agent running for {args.duration} seconds...")
        print("Waiting for market events to make pricing decisions...")
        
        # Run for specified duration
        time.sleep(args.duration)
        
    except KeyboardInterrupt:
        print("\nStopping due to keyboard interrupt...")
    
    finally:
        # Stop agent and show final stats
        final_stats = agent.stop_agent()
        print(f"\nAgent Performance Summary:")
        print(f"Market Events Processed: {final_stats.get('total_market_events', 0)}")
        print(f"Pricing Actions Sent: {final_stats.get('total_pricing_actions', 0)}")
        print(f"Model Predictions: {final_stats.get('model_predictions', 0)}")
        print(f"Fallback Decisions: {final_stats.get('fallback_decisions', 0)}")
        print(f"Avg Decision Time: {final_stats.get('avg_decision_time', 0)*1000:.2f}ms")
        if final_stats.get('confidence_scores'):
            print(f"Avg Confidence: {np.mean(list(final_stats['confidence_scores'])):.2f}")
    
    print("Done!")