"""
Test Script for Kafka Feedback Loop
Validates end-to-end system functionality with minimal test run
"""
import time
import threading
import logging
from datetime import datetime
import json

from kafka_config import KafkaHealthChecker, KafkaTopicManager, KafkaConfig, create_market_event_message, create_pricing_action_message
from kafka_config import KafkaProducerWrapper, KafkaConsumerWrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_kafka_connectivity():
    """Test basic Kafka connectivity"""
    print("=" * 50)
    print("TEST 1: Kafka Connectivity")
    print("=" * 50)
    
    # Test connection
    if not KafkaHealthChecker.check_connection():
        print("❌ FAILED: Kafka connection failed")
        return False
    
    # Create topics
    topic_manager = KafkaTopicManager()
    if not topic_manager.create_topics_if_not_exist():
        print("❌ FAILED: Topic creation failed")
        return False
    
    print("✅ PASSED: Kafka connectivity and topics OK")
    return True


def test_message_flow():
    """Test basic message flow through Kafka topics"""
    print("\n" + "=" * 50)
    print("TEST 2: Message Flow")
    print("=" * 50)
    
    messages_received = []
    
    def message_callback(topic: str, message: dict):
        messages_received.append((topic, message))
        logger.info(f"Received message from {topic}: {message.get('event_type', 'unknown')}")
    
    # Set up consumer
    consumer = KafkaConsumerWrapper(
        group_id="test_consumer",
        topics=[KafkaConfig.MARKET_EVENTS_TOPIC, KafkaConfig.PRICING_ACTIONS_TOPIC]
    )
    
    if not consumer.subscribe():
        print("❌ FAILED: Consumer subscription failed")
        return False
    
    consumer.start_consuming()
    
    # Set up producer
    producer = KafkaProducerWrapper()
    
    # Send test messages
    test_market_data = {
        'our_price': 100.0,
        'demand': 45.0,
        'competitor_prices': [95.0, 105.0],
        'inventory': 500
    }
    
    test_pricing_action = {
        'price': 102.5,
        'confidence': 0.8
    }
    
    try:
        # Send market event
        market_msg = create_market_event_message(test_market_data)
        producer.produce_message(KafkaConfig.MARKET_EVENTS_TOPIC, market_msg)
        
        # Send pricing action
        pricing_msg = create_pricing_action_message(102.5, {'confidence': 0.8})
        producer.produce_message(KafkaConfig.PRICING_ACTIONS_TOPIC, pricing_msg)
        
        # Wait for messages
        time.sleep(2)
        
        # Consume messages
        consumer.consume_messages(message_callback, timeout=3.0, max_messages=5)
        
        # Verify
        if len(messages_received) >= 2:
            print(f"✅ PASSED: Received {len(messages_received)} messages")
            for topic, msg in messages_received:
                print(f"   - {topic}: {msg.get('event_type', msg.get('action_type', 'unknown'))}")
            return True
        else:
            print(f"❌ FAILED: Expected 2+ messages, got {len(messages_received)}")
            return False
    
    finally:
        consumer.close()
        producer.close()


def test_feedback_loop_simulation():
    """Test complete feedback loop with simulated components"""
    print("\n" + "=" * 50)
    print("TEST 3: Feedback Loop Simulation")
    print("=" * 50)
    
    events_received = {'market_events': 0, 'pricing_actions': 0}
    prices = []
    
    def simulate_rl_agent():
        """Simulate RL agent that responds to market events with pricing actions"""
        consumer = KafkaConsumerWrapper(
            group_id="test_rl_agent",
            topics=[KafkaConfig.MARKET_EVENTS_TOPIC]
        )
        
        producer = KafkaProducerWrapper()
        
        def market_callback(topic: str, message: dict):
            if topic == KafkaConfig.MARKET_EVENTS_TOPIC:
                events_received['market_events'] += 1
                
                # Extract current price
                data = message.get('data', {})
                current_price = data.get('our_price', 100.0)
                demand = data.get('demand', 50.0)
                
                # Simple pricing logic
                if demand > 60:
                    new_price = current_price * 1.05  # Increase price
                elif demand < 40:
                    new_price = current_price * 0.98  # Decrease price
                else:
                    new_price = current_price
                
                # Send pricing action
                pricing_msg = create_pricing_action_message(
                    new_price, 
                    {'agent_id': 'test_agent', 'confidence': 0.7}
                )
                producer.produce_message(KafkaConfig.PRICING_ACTIONS_TOPIC, pricing_msg)
                
                prices.append(new_price)
                logger.info(f"RL Agent: Market price ${current_price:.2f}, demand {demand:.1f} → New price ${new_price:.2f}")
        
        consumer.subscribe()
        consumer.start_consuming()
        
        try:
            for _ in range(10):  # Run for 10 iterations
                consumer.consume_messages(market_callback, timeout=1.0, max_messages=5)
                time.sleep(0.5)
        finally:
            consumer.close()
            producer.close()
    
    def simulate_environment():
        """Simulate environment that responds to pricing actions with market events"""
        consumer = KafkaConsumerWrapper(
            group_id="test_environment",
            topics=[KafkaConfig.PRICING_ACTIONS_TOPIC]
        )
        
        producer = KafkaProducerWrapper()
        current_price = 100.0
        
        def pricing_callback(topic: str, message: dict):
            nonlocal current_price
            
            if topic == KafkaConfig.PRICING_ACTIONS_TOPIC:
                events_received['pricing_actions'] += 1
                
                # Update price
                new_price = message.get('price', current_price)
                current_price = new_price
                
                # Simulate demand based on price (simple elasticity)
                base_demand = 50.0
                price_effect = (100.0 / current_price) ** 1.5  # Price elasticity
                demand = base_demand * price_effect + (time.time() % 10 - 5)  # Add some variation
                demand = max(10, min(100, demand))  # Keep reasonable bounds
                
                # Create market event
                market_data = {
                    'our_price': current_price,
                    'demand': demand,
                    'competitor_prices': [95.0, 105.0, 98.0],
                    'inventory': 500,
                    'sales': min(demand, 50),  # Assume we can sell up to 50 units
                }
                
                market_msg = create_market_event_message(market_data)
                producer.produce_message(KafkaConfig.MARKET_EVENTS_TOPIC, market_msg)
                
                logger.info(f"Environment: Price ${current_price:.2f} → Demand {demand:.1f}")
        
        consumer.subscribe()
        consumer.start_consuming()
        
        try:
            for _ in range(10):  # Run for 10 iterations
                consumer.consume_messages(pricing_callback, timeout=1.0, max_messages=5)
                time.sleep(0.5)
        finally:
            consumer.close()
            producer.close()
    
    # Start simulated components in threads
    agent_thread = threading.Thread(target=simulate_rl_agent, daemon=True)
    env_thread = threading.Thread(target=simulate_environment, daemon=True)
    
    agent_thread.start()
    env_thread.start()
    
    # Kick off the loop with initial market event
    initial_producer = KafkaProducerWrapper()
    initial_market_data = {
        'our_price': 100.0,
        'demand': 55.0,
        'competitor_prices': [95.0, 105.0, 98.0],
        'inventory': 500,
        'sales': 45
    }
    initial_msg = create_market_event_message(initial_market_data)
    initial_producer.produce_message(KafkaConfig.MARKET_EVENTS_TOPIC, initial_msg)
    initial_producer.close()
    
    # Let the loop run
    logger.info("Starting feedback loop simulation...")
    time.sleep(8)  # Run for 8 seconds
    
    # Wait for threads
    agent_thread.join(timeout=2)
    env_thread.join(timeout=2)
    
    # Analyze results
    print(f"Market events processed: {events_received['market_events']}")
    print(f"Pricing actions processed: {events_received['pricing_actions']}")
    
    if len(prices) > 0:
        print(f"Price evolution: ${prices[0]:.2f} → ${prices[-1]:.2f}")
        print(f"Price changes: {len(prices)}")
    
    # Verify feedback loop worked
    if (events_received['market_events'] >= 2 and 
        events_received['pricing_actions'] >= 2 and 
        len(prices) >= 2):
        print("✅ PASSED: Feedback loop simulation successful")
        return True
    else:
        print(f"❌ FAILED: Insufficient activity for feedback loop")
        return False


def test_component_integration():
    """Test integration with actual system components"""
    print("\n" + "=" * 50)
    print("TEST 4: Component Integration")
    print("=" * 50)
    
    try:
        from kafka_market_generator import KafkaMarketDataStreamer
        from kafka_rl_agent import KafkaRLAgent
        from kafka_environment import KafkaPricingEnvironment
        
        print("✅ PASSED: All components imported successfully")
        
        # Test component initialization
        streamer = KafkaMarketDataStreamer()
        agent = KafkaRLAgent()
        environment = KafkaPricingEnvironment()
        
        print("✅ PASSED: All components initialized successfully")
        
        # Test basic component functionality
        streamer_stats = streamer.get_streaming_stats()
        agent_stats = agent.get_agent_stats()
        env_stats = environment.get_environment_stats()
        
        print("✅ PASSED: Component stats accessible")
        
        return True
        
    except ImportError as e:
        print(f"❌ FAILED: Component import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Component initialization failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary"""
    print("🔬 KAFKA FEEDBACK LOOP SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("Kafka Connectivity", test_basic_kafka_connectivity),
        ("Message Flow", test_message_flow),
        ("Feedback Loop Simulation", test_feedback_loop_simulation),
        ("Component Integration", test_component_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            logger.error(f"Test {test_name} failed with error: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"{status} {test_name}: {result}")
        
        if result == "PASSED":
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for full deployment.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check Kafka setup and dependencies.")
        return False


if __name__ == "__main__":
    # Check if Kafka is running
    print("Checking Kafka availability...")
    if not KafkaHealthChecker.wait_for_kafka(max_retries=5, retry_interval=1.0):
        print("❌ Kafka is not available. Please start Kafka on localhost:9092")
        print("\nTo start Kafka locally:")
        print("1. Download Kafka from https://kafka.apache.org/downloads")
        print("2. Start Zookeeper: bin/zookeeper-server-start.sh config/zookeeper.properties")
        print("3. Start Kafka: bin/kafka-server-start.sh config/server.properties")
        exit(1)
    
    print("✅ Kafka is available\n")
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🚀 Ready to run the full system!")
        print("Use: python kafka_orchestrator.py --duration 60 --fast-mode")
    else:
        exit(1)