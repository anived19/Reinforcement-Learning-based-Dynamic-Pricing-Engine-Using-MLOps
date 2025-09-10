import json
import time
import logging
from typing import Dict, Any, Optional, Callable
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
from datetime import datetime
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KafkaConfig:
    """Kafka configuration settings"""
    
    BOOTSTRAP_SERVERS = 'localhost:9092'
    
    # Topic names
    MARKET_EVENTS_TOPIC = 'market_events'
    PRICING_ACTIONS_TOPIC = 'pricing_actions'
    SYSTEM_METRICS_TOPIC = 'system_metrics'
    
    # Producer config
    PRODUCER_CONFIG = {
        'bootstrap.servers': BOOTSTRAP_SERVERS,
        'client.id': 'pricing_system_producer',
        'acks': 'all',  # Wait for all replicas to acknowledge
        'retries': 3,
        'batch.size': 16384,
        'linger.ms': 10,
        'compression.type': 'snappy'
    }
    
    # Consumer config base
    CONSUMER_CONFIG_BASE = {
        'bootstrap.servers': BOOTSTRAP_SERVERS,
        'auto.offset.reset': 'latest',  # Start from latest messages
        'enable.auto.commit': False,    # Manual commit for reliability
        'session.timeout.ms': 30000,
        'heartbeat.interval.ms': 10000,
    }


class KafkaTopicManager:
    """Manages Kafka topics creation and validation"""
    
    def __init__(self):
        self.admin_client = AdminClient({'bootstrap.servers': KafkaConfig.BOOTSTRAP_SERVERS})
    
    def create_topics_if_not_exist(self) -> bool:
        """Create necessary topics if they don't exist"""
        try:
            topics_to_create = [
                NewTopic(KafkaConfig.MARKET_EVENTS_TOPIC, num_partitions=3, replication_factor=1),
                NewTopic(KafkaConfig.PRICING_ACTIONS_TOPIC, num_partitions=3, replication_factor=1),
                NewTopic(KafkaConfig.SYSTEM_METRICS_TOPIC, num_partitions=1, replication_factor=1)
            ]
            
            # Get existing topics
            metadata = self.admin_client.list_topics(timeout=10)
            existing_topics = set(metadata.topics.keys())
            
            # Filter out topics that already exist
            topics_to_create = [
                topic for topic in topics_to_create 
                if topic.topic not in existing_topics
            ]
            
            if topics_to_create:
                logger.info(f"Creating topics: {[t.topic for t in topics_to_create]}")
                fs = self.admin_client.create_topics(topics_to_create)
                
                # Wait for operation to complete
                for topic, f in fs.items():
                    try:
                        f.result()  # The result itself is None
                        logger.info(f"Topic {topic} created successfully")
                    except KafkaException as e:
                        logger.error(f"Failed to create topic {topic}: {e}")
                        return False
            else:
                logger.info("All required topics already exist")
            
            return True
            
        except Exception as e:
            logger.error(f"Error managing topics: {e}")
            return False


class KafkaProducerWrapper:
    """Thread-safe Kafka producer wrapper"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or KafkaConfig.PRODUCER_CONFIG
        self.producer = Producer(self.config)
        self._lock = threading.Lock()
        
    def produce_message(self, topic: str, message: Dict[str, Any], key: str = None) -> bool:
        """Produce a message to specified topic"""
        try:
            with self._lock:
                # Add timestamp if not present
                if 'timestamp' not in message:
                    message['timestamp'] = datetime.utcnow().isoformat()
                
                # Serialize message to JSON
                message_json = json.dumps(message, default=str)
                
                # Produce message
                self.producer.produce(
                    topic=topic,
                    key=key,
                    value=message_json,
                    callback=self._delivery_callback
                )
                
                # Trigger delivery
                self.producer.poll(0)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to produce message to {topic}: {e}")
            return False
    
    def _delivery_callback(self, err, msg):
        """Callback for message delivery confirmation"""
        if err:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')
    
    def flush(self, timeout: float = 10.0):
        """Wait for all messages to be delivered"""
        self.producer.flush(timeout)
    
    def close(self):
        """Close the producer"""
        self.flush()
        self.producer = None


class KafkaConsumerWrapper:
    """Thread-safe Kafka consumer wrapper"""
    
    def __init__(self, group_id: str, topics: list, config: Dict[str, Any] = None):
        consumer_config = KafkaConfig.CONSUMER_CONFIG_BASE.copy()
        consumer_config['group.id'] = group_id
        
        if config:
            consumer_config.update(config)
        
        self.consumer = Consumer(consumer_config)
        self.topics = topics
        self.running = False
        self._lock = threading.Lock()
        
    def subscribe(self):
        """Subscribe to topics"""
        try:
            self.consumer.subscribe(self.topics)
            logger.info(f"Subscribed to topics: {self.topics}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to topics: {e}")
            return False
    
    def consume_messages(self, 
                        callback: Callable[[str, Dict[str, Any]], None], 
                        timeout: float = 1.0,
                        max_messages: int = None) -> int:
        """
        Consume messages and call callback for each message
        
        Args:
            callback: Function to call for each message (topic, message_dict)
            timeout: Timeout for polling in seconds
            max_messages: Maximum number of messages to consume (None for unlimited)
        
        Returns:
            Number of messages processed
        """
        if not self.running:
            return 0
            
        messages_processed = 0
        
        try:
            while self.running and (max_messages is None or messages_processed < max_messages):
                msg = self.consumer.poll(timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f'Reached end of partition for {msg.topic()}')
                        continue
                    else:
                        logger.error(f'Consumer error: {msg.error()}')
                        continue
                
                try:
                    # Parse message
                    message_data = json.loads(msg.value().decode('utf-8'))
                    topic = msg.topic()
                    
                    # Call user-provided callback
                    callback(topic, message_data)
                    
                    # Commit offset
                    self.consumer.commit(msg)
                    messages_processed += 1
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        
        return messages_processed
    
    def start_consuming(self):
        """Start the consumer"""
        with self._lock:
            self.running = True
        logger.info("Consumer started")
    
    def stop_consuming(self):
        """Stop the consumer"""
        with self._lock:
            self.running = False
        logger.info("Consumer stopped")
    
    def close(self):
        """Close the consumer"""
        self.stop_consuming()
        self.consumer.close()


class KafkaHealthChecker:
    """Check Kafka cluster health and connectivity"""
    
    @staticmethod
    def check_connection() -> bool:
        """Check if Kafka is accessible"""
        try:
            admin_client = AdminClient({'bootstrap.servers': KafkaConfig.BOOTSTRAP_SERVERS})
            metadata = admin_client.list_topics(timeout=5)
            logger.info(f"Kafka connection successful. Available topics: {list(metadata.topics.keys())}")
            return True
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            return False
    
    @staticmethod
    def wait_for_kafka(max_retries: int = 30, retry_interval: float = 2.0) -> bool:
        """Wait for Kafka to become available"""
        for attempt in range(max_retries):
            if KafkaHealthChecker.check_connection():
                return True
            
            logger.info(f"Kafka not ready, attempt {attempt + 1}/{max_retries}. Retrying in {retry_interval}s...")
            time.sleep(retry_interval)
        
        logger.error("Kafka failed to become available within timeout period")
        return False


# Utility functions for message creation
def create_market_event_message(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized market event message"""
    return {
        'event_type': 'market_update',
        'timestamp': datetime.utcnow().isoformat(),
        'data': market_data
    }


def create_pricing_action_message(price: float, agent_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create standardized pricing action message"""
    return {
        'action_type': 'price_update',
        'timestamp': datetime.utcnow().isoformat(),
        'price': price,
        'agent_info': agent_info or {}
    }


def create_system_metrics_message(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized system metrics message"""
    return {
        'metrics_type': 'system_performance',
        'timestamp': datetime.utcnow().isoformat(),
        'metrics': metrics
    }


# Example usage and testing
if __name__ == "__main__":
    # Test Kafka setup
    print("Testing Kafka setup...")
    
    # Check connection
    if not KafkaHealthChecker.wait_for_kafka(max_retries=5, retry_interval=1.0):
        print("Failed to connect to Kafka. Make sure Kafka is running on localhost:9092")
        exit(1)
    
    # Create topics
    topic_manager = KafkaTopicManager()
    if not topic_manager.create_topics_if_not_exist():
        print("Failed to create topics")
        exit(1)
    
    # Test producer
    producer = KafkaProducerWrapper()
    
    # Test consumer
    def test_callback(topic: str, message: Dict[str, Any]):
        print(f"Received from {topic}: {message}")
    
    consumer = KafkaConsumerWrapper(
        group_id="test_group",
        topics=[KafkaConfig.MARKET_EVENTS_TOPIC, KafkaConfig.PRICING_ACTIONS_TOPIC]
    )
    
    if not consumer.subscribe():
        print("Failed to subscribe to topics")
        exit(1)
    
    # Test message flow
    print("\nTesting message flow...")
    consumer.start_consuming()
    
    # Send test messages
    test_market_data = {
        'demand': 45.5,
        'competitor_prices': [95.0, 105.0, 98.0],
        'inventory': 500,
        'external_events': []
    }
    
    test_pricing_action = {
        'price': 102.5,
        'confidence': 0.8
    }
    
    # Produce messages
    producer.produce_message(
        KafkaConfig.MARKET_EVENTS_TOPIC, 
        create_market_event_message(test_market_data)
    )
    
    producer.produce_message(
        KafkaConfig.PRICING_ACTIONS_TOPIC,
        create_pricing_action_message(102.5, {'confidence': 0.8})
    )
    
    # Consume messages
    messages_processed = consumer.consume_messages(test_callback, timeout=2.0, max_messages=5)
    print(f"Processed {messages_processed} messages")
    
    # Cleanup
    producer.close()
    consumer.close()
    
    print("Kafka setup test completed successfully!")