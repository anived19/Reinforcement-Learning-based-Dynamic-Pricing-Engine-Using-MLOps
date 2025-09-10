"""
Kafka-Enabled Market Data Generator
Integrates with the original data_generation.py to stream market events to Kafka
"""
import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from data_generation import MarketDataGenerator, RealTimeMarketSimulator
from kafka_config import (
    KafkaConfig, KafkaProducerWrapper, KafkaConsumerWrapper,
    create_market_event_message, create_system_metrics_message,
    KafkaHealthChecker, KafkaTopicManager
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KafkaMarketDataStreamer:
    """Streams market data to Kafka topics in real-time"""
    
    def __init__(self, product_config: Dict = None, streaming_config: Dict = None):
        self.product_config = product_config or {
            'name': 'Gaming Laptop',
            'base_price': 1200.0,
            'cost': 800.0,
            'initial_inventory': 1000
        }
        
        self.streaming_config = streaming_config or {
            'interval_seconds': 5.0,  # How often to generate market data
            'batch_size': 1,          # Number of events per batch
            'auto_pricing': True,     # Whether to use auto-pricing or wait for RL agent
            'price_volatility': 0.05  # Random price variation for auto-pricing
        }
        
        # Initialize market simulator
        self.market_simulator = RealTimeMarketSimulator(self.product_config)
        
        # Kafka components
        self.producer = KafkaProducerWrapper()
        self.pricing_consumer = None
        
        # State management
        self.running = False
        self.current_price = self.product_config['base_price']
        self.received_pricing_actions = []
        self.market_stats = {
            'total_events_sent': 0,
            'total_pricing_actions_received': 0,
            'avg_processing_time': 0.0,
            'last_market_data': None
        }
        
        # Threading
        self.producer_thread = None
        self.consumer_thread = None
        
    def setup_kafka_consumer(self):
        """Set up Kafka consumer for pricing actions"""
        def pricing_action_callback(topic: str, message: Dict[str, Any]):
            if topic == KafkaConfig.PRICING_ACTIONS_TOPIC:
                self._handle_pricing_action(message)
        
        self.pricing_consumer = KafkaConsumerWrapper(
            group_id="market_data_generator",
            topics=[KafkaConfig.PRICING_ACTIONS_TOPIC]
        )
        
        if not self.pricing_consumer.subscribe():
            raise Exception("Failed to subscribe to pricing actions topic")
        
        logger.info("Kafka consumer for pricing actions set up successfully")
    
    def _handle_pricing_action(self, message: Dict[str, Any]):
        """Handle incoming pricing action from RL agent"""
        try:
            action_type = message.get('action_type')
            if action_type == 'price_update':
                new_price = message.get('price')
                if new_price and isinstance(new_price, (int, float)):
                    self.current_price = float(new_price)
                    self.received_pricing_actions.append(message)
                    self.market_stats['total_pricing_actions_received'] += 1
                    
                    logger.info(f"Received price update: ${new_price:.2f} from RL agent")
                else:
                    logger.warning(f"Invalid price in action message: {new_price}")
            else:
                logger.warning(f"Unknown action type: {action_type}")
        
        except Exception as e:
            logger.error(f"Error handling pricing action: {e}")
    
    def _generate_market_event(self) -> Dict[str, Any]:
        """Generate a single market event"""
        start_time = time.time()
        
        # Use current price (either from RL agent or auto-pricing)
        if self.streaming_config['auto_pricing'] and len(self.received_pricing_actions) == 0:
            # Auto-pricing: add some random variation
            import numpy as np
            price_change = np.random.normal(0, self.streaming_config['price_volatility'])
            self.current_price = self.current_price * (1 + price_change)
            self.current_price = max(self.current_price, self.product_config['cost'] * 1.1)
        
        # Generate market data
        market_data = self.market_simulator.step_simulation(self.current_price)
        
        # Add processing metadata
        processing_time = time.time() - start_time
        self.market_stats['avg_processing_time'] = (
            (self.market_stats['avg_processing_time'] * self.market_stats['total_events_sent'] + processing_time) /
            (self.market_stats['total_events_sent'] + 1)
        )
        self.market_stats['last_market_data'] = market_data
        
        return market_data
    
    def _publish_market_event(self, market_data: Dict[str, Any]) -> bool:
        """Publish market event to Kafka"""
        try:
            # Create standardized message
            message = create_market_event_message(market_data)
            
            # Add streaming metadata
            message['producer_id'] = 'market_data_generator'
            message['sequence_number'] = self.market_stats['total_events_sent'] + 1
            
            # Publish to Kafka
            success = self.producer.produce_message(
                topic=KafkaConfig.MARKET_EVENTS_TOPIC,
                message=message,
                key=f"market_{int(time.time())}"
            )
            
            if success:
                self.market_stats['total_events_sent'] += 1
                logger.debug(f"Published market event #{self.market_stats['total_events_sent']}")
            else:
                logger.error("Failed to publish market event")
            
            return success
            
        except Exception as e:
            logger.error(f"Error publishing market event: {e}")
            return False
    
    def _publish_system_metrics(self):
        """Publish system performance metrics"""
        try:
            metrics = {
                'events_sent_per_second': (
                    self.market_stats['total_events_sent'] / 
                    max(1, time.time() - self.start_time)
                ),
                'total_events_sent': self.market_stats['total_events_sent'],
                'total_pricing_actions_received': self.market_stats['total_pricing_actions_received'],
                'avg_processing_time_ms': self.market_stats['avg_processing_time'] * 1000,
                'current_price': self.current_price,
                'market_simulator_stats': self.market_simulator.data_generator.get_market_statistics()
            }
            
            message = create_system_metrics_message(metrics)
            self.producer.produce_message(
                topic=KafkaConfig.SYSTEM_METRICS_TOPIC,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error publishing system metrics: {e}")
    
    def _producer_loop(self):
        """Main producer loop - generates and publishes market events"""
        logger.info("Market data producer loop started")
        
        while self.running:
            try:
                # Generate market event
                market_data = self._generate_market_event()
                
                # Publish to Kafka
                self._publish_market_event(market_data)
                
                # Publish system metrics every 10 events
                if self.market_stats['total_events_sent'] % 10 == 0:
                    self._publish_system_metrics()
                
                # Log progress
                if self.market_stats['total_events_sent'] % 20 == 0:
                    logger.info(f"Published {self.market_stats['total_events_sent']} market events")
                
                # Wait for next iteration
                time.sleep(self.streaming_config['interval_seconds'])
                
            except Exception as e:
                logger.error(f"Error in producer loop: {e}")
                time.sleep(1)  # Brief pause before retrying
        
        logger.info("Market data producer loop stopped")
    
    def _consumer_loop(self):
        """Consumer loop - listens for pricing actions"""
        logger.info("Pricing action consumer loop started")
        
        self.pricing_consumer.start_consuming()
        
        while self.running:
            try:
                # Process messages with timeout
                self.pricing_consumer.consume_messages(
                    callback=lambda topic, msg: None,  # Callback handled in setup
                    timeout=1.0,
                    max_messages=10
                )
                
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                time.sleep(1)
        
        logger.info("Pricing action consumer loop stopped")
    
    def start_streaming(self):
        """Start the market data streaming system"""
        if self.running:
            logger.warning("Streaming system already running")
            return
        
        logger.info("Starting Kafka market data streaming system...")
        
        # Setup Kafka consumer
        self.setup_kafka_consumer()
        
        # Initialize market simulator
        initial_market_data = self.market_simulator.start_simulation(self.current_price)
        logger.info(f"Market simulator initialized with base price: ${self.current_price}")
        
        # Start streaming
        self.running = True
        self.start_time = time.time()
        
        # Start producer thread
        self.producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.producer_thread.start()
        
        # Start consumer thread
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()
        
        logger.info("Market data streaming system started successfully")
        
        # Publish initial event
        self._publish_market_event(initial_market_data)
    
    def stop_streaming(self):
        """Stop the streaming system"""
        if not self.running:
            logger.warning("Streaming system not running")
            return
        
        logger.info("Stopping market data streaming system...")
        
        self.running = False
        
        # Wait for threads to complete
        if self.producer_thread and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=5.0)
        
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5.0)
        
        # Close Kafka components
        if self.pricing_consumer:
            self.pricing_consumer.close()
        
        # Flush and close producer
        self.producer.flush(timeout=5.0)
        self.producer.close()
        
        # Get final stats
        final_stats = self.market_simulator.stop_simulation()
        
        logger.info("Market data streaming system stopped")
        return {**self.market_stats, **final_stats}
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        current_time = time.time()
        runtime = current_time - getattr(self, 'start_time', current_time)
        
        return {
            **self.market_stats,
            'runtime_seconds': runtime,
            'events_per_second': self.market_stats['total_events_sent'] / max(1, runtime),
            'current_price': self.current_price,
            'system_running': self.running,
            'last_event_time': datetime.utcnow().isoformat()
        }


class KafkaMarketDataMonitor:
    """Monitor market data stream and display statistics"""
    
    def __init__(self):
        self.consumer = KafkaConsumerWrapper(
            group_id="market_monitor",
            topics=[
                KafkaConfig.MARKET_EVENTS_TOPIC,
                KafkaConfig.PRICING_ACTIONS_TOPIC,
                KafkaConfig.SYSTEM_METRICS_TOPIC
            ]
        )
        self.stats = {
            'market_events': 0,
            'pricing_actions': 0,
            'system_metrics': 0,
            'last_prices': [],
            'last_demands': [],
            'start_time': None
        }
        self.running = False
    
    def _monitor_callback(self, topic: str, message: Dict[str, Any]):
        """Handle incoming monitoring messages"""
        if self.stats['start_time'] is None:
            self.stats['start_time'] = time.time()
        
        if topic == KafkaConfig.MARKET_EVENTS_TOPIC:
            self.stats['market_events'] += 1
            if 'data' in message:
                data = message['data']
                if 'our_price' in data:
                    self.stats['last_prices'].append(data['our_price'])
                    self.stats['last_prices'] = self.stats['last_prices'][-10:]  # Keep last 10
                
                if 'demand' in data:
                    self.stats['last_demands'].append(data['demand'])
                    self.stats['last_demands'] = self.stats['last_demands'][-10:]  # Keep last 10
        
        elif topic == KafkaConfig.PRICING_ACTIONS_TOPIC:
            self.stats['pricing_actions'] += 1
        
        elif topic == KafkaConfig.SYSTEM_METRICS_TOPIC:
            self.stats['system_metrics'] += 1
        
        # Print periodic updates
        total_messages = sum([
            self.stats['market_events'],
            self.stats['pricing_actions'],
            self.stats['system_metrics']
        ])
        
        if total_messages % 10 == 0:
            self.print_stats()
    
    def print_stats(self):
        """Print current monitoring statistics"""
        runtime = time.time() - (self.stats['start_time'] or time.time())
        
        print(f"\n{'='*50}")
        print(f"MARKET DATA STREAM MONITOR")
        print(f"{'='*50}")
        print(f"Runtime: {runtime:.1f}s")
        print(f"Market Events: {self.stats['market_events']}")
        print(f"Pricing Actions: {self.stats['pricing_actions']}")
        print(f"System Metrics: {self.stats['system_metrics']}")
        
        if self.stats['last_prices']:
            avg_price = sum(self.stats['last_prices']) / len(self.stats['last_prices'])
            print(f"Avg Recent Price: ${avg_price:.2f}")
        
        if self.stats['last_demands']:
            avg_demand = sum(self.stats['last_demands']) / len(self.stats['last_demands'])
            print(f"Avg Recent Demand: {avg_demand:.1f}")
        
        print(f"{'='*50}\n")
    
    def start_monitoring(self):
        """Start monitoring the stream"""
        if not self.consumer.subscribe():
            logger.error("Failed to subscribe to monitoring topics")
            return False
        
        self.running = True
        self.consumer.start_consuming()
        
        logger.info("Started monitoring market data stream...")
        
        try:
            while self.running:
                self.consumer.consume_messages(
                    callback=self._monitor_callback,
                    timeout=2.0,
                    max_messages=50
                )
        except KeyboardInterrupt:
            logger.info("Monitor interrupted by user")
        
        return True
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        self.consumer.close()
        logger.info("Market data monitoring stopped")


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Market Data Streamer')
    parser.add_argument('--mode', choices=['stream', 'monitor'], default='stream',
                      help='Run in streaming or monitoring mode')
    parser.add_argument('--duration', type=int, default=60,
                      help='How long to run (seconds)')
    parser.add_argument('--interval', type=float, default=3.0,
                      help='Interval between market events (seconds)')
    
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
    
    if args.mode == 'stream':
        # Streaming mode
        print(f"Starting market data streaming for {args.duration} seconds...")
        
        config = {
            'interval_seconds': args.interval,
            'auto_pricing': True,  # Use auto-pricing since no RL agent yet
            'price_volatility': 0.02
        }
        
        streamer = KafkaMarketDataStreamer(streaming_config=config)
        
        try:
            streamer.start_streaming()
            
            # Run for specified duration
            time.sleep(args.duration)
            
        except KeyboardInterrupt:
            print("\nStopping due to keyboard interrupt...")
        
        finally:
            # Stop streaming and show final stats
            final_stats = streamer.stop_streaming()
            print(f"\nFinal Statistics:")
            print(f"Total Events Sent: {final_stats.get('total_events_sent', 0)}")
            print(f"Total Profit: ${final_stats.get('total_profit', 0):.2f}")
            print(f"Avg Processing Time: {final_stats.get('avg_processing_time', 0)*1000:.2f}ms")
    
    elif args.mode == 'monitor':
        # Monitoring mode
        print(f"Starting market data monitoring for {args.duration} seconds...")
        
        monitor = KafkaMarketDataMonitor()
        
        def stop_monitor():
            time.sleep(args.duration)
            monitor.stop_monitoring()
        
        # Start stop timer in background
        stop_thread = threading.Thread(target=stop_monitor, daemon=True)
        stop_thread.start()
        
        # Start monitoring (blocking)
        monitor.start_monitoring()
        
        print("Monitoring completed!")
    
    print("Done!")