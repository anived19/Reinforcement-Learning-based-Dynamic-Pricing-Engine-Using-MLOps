"""
Kafka Feedback Loop Orchestrator
Coordinates all components of the dynamic pricing system with real-time Kafka messaging
"""
import time
import threading
import signal
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import argparse

from kafka_config import KafkaHealthChecker, KafkaTopicManager, KafkaConfig
from kafka_market_generator import KafkaMarketDataStreamer, KafkaMarketDataMonitor
from kafka_rl_agent import KafkaRLAgent
from kafka_environment import KafkaPricingEnvironment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemOrchestrator:
    """Orchestrates the entire dynamic pricing feedback loop system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # System components
        self.market_streamer = None
        self.rl_agent = None
        self.pricing_environment = None
        self.monitor = None
        
        # System state
        self.components_running = []
        self.system_running = False
        self.start_time = None
        self.shutdown_requested = False
        
        # Statistics
        self.system_stats = {
            'total_runtime': 0.0,
            'components_started': [],
            'components_stopped': [],
            'errors': [],
            'last_health_check': None
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            'mode': 'full_loop',  # Options: 'full_loop', 'market_only', 'agent_only', 'env_only'
            'duration': 300,      # Runtime in seconds
            'components': {
                'market_streamer': {
                    'enabled': True,
                    'config': {
                        'interval_seconds': 6.0,
                        'auto_pricing': False,  # Let RL agent handle pricing
                        'price_volatility': 0.02
                    }
                },
                'rl_agent': {
                    'enabled': True,
                    'model_path': None,  # Will use fallback pricing if no model
                    'config': {
                        'decision_interval': 8.0,
                        'confidence_threshold': 0.2,
                        'max_price_change': 0.12
                    }
                },
                'pricing_environment': {
                    'enabled': True,
                    'config': {
                        'update_interval': 5.0,
                        'auto_advance': True
                    }
                },
                'monitor': {
                    'enabled': True
                }
            },
            'health_check_interval': 30.0,  # Seconds between health checks
            'startup_delay': 2.0,           # Delay between component starts
            'shutdown_timeout': 10.0        # Max time to wait for component shutdown
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def _validate_kafka_setup(self) -> bool:
        """Validate Kafka setup and create topics if needed"""
        logger.info("Validating Kafka setup...")
        
        # Check Kafka connection
        if not KafkaHealthChecker.wait_for_kafka(max_retries=15, retry_interval=2.0):
            logger.error("Failed to connect to Kafka")
            return False
        
        # Create topics
        topic_manager = KafkaTopicManager()
        if not topic_manager.create_topics_if_not_exist():
            logger.error("Failed to create Kafka topics")
            return False
        
        logger.info("Kafka setup validated successfully")
        return True
    
    def _start_market_streamer(self):
        """Start market data streamer component"""
        if not self.config['components']['market_streamer']['enabled']:
            return
        
        try:
            logger.info("Starting Market Data Streamer...")
            
            streamer_config = self.config['components']['market_streamer']['config']
            self.market_streamer = KafkaMarketDataStreamer(streaming_config=streamer_config)
            self.market_streamer.start_streaming()
            
            self.components_running.append('market_streamer')
            self.system_stats['components_started'].append({
                'component': 'market_streamer',
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info("Market Data Streamer started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Market Data Streamer: {e}")
            self.system_stats['errors'].append({
                'component': 'market_streamer',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            raise
    
    def _start_rl_agent(self):
        """Start RL agent component"""
        if not self.config['components']['rl_agent']['enabled']:
            return
        
        try:
            logger.info("Starting RL Agent...")
            
            agent_config = self.config['components']['rl_agent']['config']
            model_path = self.config['components']['rl_agent']['model_path']
            
            self.rl_agent = KafkaRLAgent(model_path=model_path, agent_config=agent_config)
            self.rl_agent.start_agent()
            
            self.components_running.append('rl_agent')
            self.system_stats['components_started'].append({
                'component': 'rl_agent',
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info("RL Agent started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start RL Agent: {e}")
            self.system_stats['errors'].append({
                'component': 'rl_agent',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            raise
    
    def _start_pricing_environment(self):
        """Start pricing environment component"""
        if not self.config['components']['pricing_environment']['enabled']:
            return
        
        try:
            logger.info("Starting Pricing Environment...")
            
            env_config = {
                'base_price': 110.0,
                'base_cost': 75.0,
                'price_elasticity': -1.6,
                'competitor_reactivity': 0.35
            }
            
            kafka_config = self.config['components']['pricing_environment']['config']
            kafka_config['environment_id'] = 'orchestrated_pricing_env'
            
            self.pricing_environment = KafkaPricingEnvironment(
                env_config=env_config,
                kafka_config=kafka_config
            )
            self.pricing_environment.start_environment()
            
            self.components_running.append('pricing_environment')
            self.system_stats['components_started'].append({
                'component': 'pricing_environment',
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info("Pricing Environment started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Pricing Environment: {e}")
            self.system_stats['errors'].append({
                'component': 'pricing_environment',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            raise
    
    def _start_monitor(self):
        """Start system monitor component"""
        if not self.config['components']['monitor']['enabled']:
            return
        
        try:
            logger.info("Starting System Monitor...")
            
            self.monitor = KafkaMarketDataMonitor()
            
            # Start monitor in separate thread
            monitor_thread = threading.Thread(
                target=self._run_monitor,
                daemon=True
            )
            monitor_thread.start()
            
            self.components_running.append('monitor')
            self.system_stats['components_started'].append({
                'component': 'monitor',
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info("System Monitor started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start System Monitor: {e}")
            self.system_stats['errors'].append({
                'component': 'monitor',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def _run_monitor(self):
        """Run monitor in background thread"""
        try:
            self.monitor.start_monitoring()
        except Exception as e:
            logger.error(f"Monitor error: {e}")
    
    def _perform_health_check(self):
        """Perform system health check"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'components_running': len(self.components_running),
            'expected_components': sum(1 for comp in self.config['components'].values() if comp['enabled']),
            'system_healthy': True,
            'component_stats': {}
        }
        
        try:
            # Check market streamer
            if self.market_streamer and 'market_streamer' in self.components_running:
                streamer_stats = self.market_streamer.get_streaming_stats()
                health_status['component_stats']['market_streamer'] = {
                    'events_sent': streamer_stats.get('total_events_sent', 0),
                    'running': streamer_stats.get('system_running', False)
                }
            
            # Check RL agent
            if self.rl_agent and 'rl_agent' in self.components_running:
                agent_stats = self.rl_agent.get_agent_stats()
                health_status['component_stats']['rl_agent'] = {
                    'actions_sent': agent_stats.get('total_pricing_actions', 0),
                    'events_received': agent_stats.get('total_market_events', 0),
                    'running': agent_stats.get('agent_running', False)
                }
            
            # Check pricing environment
            if self.pricing_environment and 'pricing_environment' in self.components_running:
                env_stats = self.pricing_environment.get_environment_stats()
                health_status['component_stats']['pricing_environment'] = {
                    'market_updates': env_stats.get('total_market_updates', 0),
                    'pricing_actions': env_stats.get('total_pricing_actions', 0),
                    'running': env_stats.get('environment_running', False)
                }
            
            # Overall health assessment
            if len(self.components_running) < health_status['expected_components']:
                health_status['system_healthy'] = False
                logger.warning(f"System health warning: {len(self.components_running)}/{health_status['expected_components']} components running")
            
            self.system_stats['last_health_check'] = health_status
            
            logger.info(f"Health check completed: {len(self.components_running)}/{health_status['expected_components']} components healthy")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['system_healthy'] = False
            health_status['error'] = str(e)
    
    def start_system(self):
        """Start the entire system"""
        if self.system_running:
            logger.warning("System already running")
            return False
        
        logger.info("="*60)
        logger.info("STARTING DYNAMIC PRICING RL SYSTEM")
        logger.info("="*60)
        
        try:
            # Validate Kafka setup
            if not self._validate_kafka_setup():
                return False
            
            self.system_running = True
            self.start_time = time.time()
            
            startup_delay = self.config['startup_delay']
            
            # Start components in order
            component_start_order = ['pricing_environment', 'market_streamer', 'rl_agent', 'monitor']
            
            for component in component_start_order:
                if self.shutdown_requested:
                    break
                
                if component == 'market_streamer':
                    self._start_market_streamer()
                elif component == 'rl_agent':
                    self._start_rl_agent()
                elif component == 'pricing_environment':
                    self._start_pricing_environment()
                elif component == 'monitor':
                    self._start_monitor()
                
                # Delay between component starts
                if component != component_start_order[-1] and not self.shutdown_requested:
                    logger.info(f"Waiting {startup_delay}s before starting next component...")
                    time.sleep(startup_delay)
            
            if not self.shutdown_requested:
                logger.info("="*60)
                logger.info("SYSTEM STARTUP COMPLETE")
                logger.info(f"Started {len(self.components_running)} components successfully")
                logger.info("="*60)
                
                # Initial health check
                self._perform_health_check()
                
                return True
            else:
                logger.info("System startup cancelled due to shutdown request")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.system_stats['errors'].append({
                'component': 'system',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            return False
    
    def run_system(self, duration: int = None):
        """Run the system for specified duration"""
        duration = duration or self.config['duration']
        health_check_interval = self.config['health_check_interval']
        
        logger.info(f"Running system for {duration} seconds...")
        logger.info("System feedback loop active. Components communicating via Kafka.")
        logger.info("-" * 60)
        
        try:
            end_time = time.time() + duration
            last_health_check = time.time()
            
            while time.time() < end_time and not self.shutdown_requested:
                current_time = time.time()
                
                # Perform periodic health checks
                if current_time - last_health_check >= health_check_interval:
                    self._perform_health_check()
                    last_health_check = current_time
                
                # Sleep briefly
                time.sleep(2.0)
            
            if self.shutdown_requested:
                logger.info("System run interrupted by shutdown request")
            else:
                logger.info(f"System run completed after {duration} seconds")
                
        except KeyboardInterrupt:
            logger.info("System run interrupted by user")
            self.shutdown_requested = True
        except Exception as e:
            logger.error(f"Error during system run: {e}")
            self.system_stats['errors'].append({
                'component': 'system_run',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def stop_system(self):
        """Stop all system components gracefully"""
        if not self.system_running:
            logger.warning("System not running")
            return {}
        
        logger.info("="*60)
        logger.info("STOPPING DYNAMIC PRICING RL SYSTEM")
        logger.info("="*60)
        
        shutdown_timeout = self.config['shutdown_timeout']
        final_stats = {}
        
        # Stop components in reverse order
        components_to_stop = list(reversed(self.components_running))
        
        for component in components_to_stop:
            try:
                logger.info(f"Stopping {component}...")
                
                if component == 'market_streamer' and self.market_streamer:
                    stats = self.market_streamer.stop_streaming()
                    final_stats['market_streamer'] = stats
                    
                elif component == 'rl_agent' and self.rl_agent:
                    stats = self.rl_agent.stop_agent()
                    final_stats['rl_agent'] = stats
                    
                elif component == 'pricing_environment' and self.pricing_environment:
                    stats = self.pricing_environment.stop_environment()
                    final_stats['pricing_environment'] = stats
                    
                elif component == 'monitor' and self.monitor:
                    self.monitor.stop_monitoring()
                    final_stats['monitor'] = {'status': 'stopped'}
                
                self.system_stats['components_stopped'].append({
                    'component': component,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                logger.info(f"{component} stopped successfully")
                
            except Exception as e:
                logger.error(f"Error stopping {component}: {e}")
                self.system_stats['errors'].append({
                    'component': component,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        # Calculate total runtime
        if self.start_time:
            self.system_stats['total_runtime'] = time.time() - self.start_time
        
        self.system_running = False
        self.components_running.clear()
        
        logger.info("="*60)
        logger.info("SYSTEM SHUTDOWN COMPLETE")
        logger.info("="*60)
        
        return {
            'system_stats': self.system_stats,
            'component_stats': final_stats
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        runtime = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            'system_status': {
                'running': self.system_running,
                'runtime_seconds': runtime,
                'components_running': len(self.components_running),
                'components': self.components_running.copy(),
                'errors_count': len(self.system_stats['errors'])
            },
            'component_performance': {},
            'system_stats': self.system_stats.copy(),
            'last_health_check': self.system_stats.get('last_health_check'),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Get component-specific stats
        try:
            if self.market_streamer:
                summary['component_performance']['market_streamer'] = self.market_streamer.get_streaming_stats()
            
            if self.rl_agent:
                summary['component_performance']['rl_agent'] = self.rl_agent.get_agent_stats()
            
            if self.pricing_environment:
                summary['component_performance']['pricing_environment'] = self.pricing_environment.get_environment_stats()
        
        except Exception as e:
            logger.error(f"Error collecting component stats: {e}")
        
        return summary


def print_system_banner():
    """Print system startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                     DYNAMIC PRICING RL SYSTEM                               ║
    ║                          Kafka Feedback Loop                                ║
    ║                                                                              ║
    ║  Components:                                                                 ║
    ║    • Market Data Streamer  → Generates market events                        ║
    ║    • RL Agent             → Makes pricing decisions                         ║
    ║    • Pricing Environment  → Simulates market response                       ║
    ║    • System Monitor       → Tracks performance                              ║
    ║                                                                              ║
    ║  Data Flow:                                                                  ║
    ║    Market Events → Kafka → RL Agent → Kafka → Environment → Kafka → ...    ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Dynamic Pricing RL System with Kafka Feedback Loop',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full system for 5 minutes
  python kafka_orchestrator.py --duration 300

  # Run with custom model
  python kafka_orchestrator.py --model-path models/best_model --duration 180

  # Run only market streamer and environment (no RL agent)
  python kafka_orchestrator.py --disable-agent --duration 120

  # Quick test run
  python kafka_orchestrator.py --duration 60 --fast-mode
        """
    )
    
    parser.add_argument('--duration', type=int, default=300,
                      help='System runtime in seconds (default: 300)')
    
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to trained RL model (optional)')
    
    parser.add_argument('--disable-streamer', action='store_true',
                      help='Disable market data streamer')
    
    parser.add_argument('--disable-agent', action='store_true',
                      help='Disable RL agent')
    
    parser.add_argument('--disable-environment', action='store_true',
                      help='Disable pricing environment')
    
    parser.add_argument('--disable-monitor', action='store_true',
                      help='Disable system monitor')
    
    parser.add_argument('--fast-mode', action='store_true',
                      help='Use faster intervals for testing')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Print banner
    print_system_banner()
    
    # Create system configuration
    config = {
        'duration': args.duration,
        'components': {
            'market_streamer': {
                'enabled': not args.disable_streamer,
                'config': {
                    'interval_seconds': 3.0 if args.fast_mode else 6.0,
                    'auto_pricing': args.disable_agent,  # Use auto-pricing if no RL agent
                    'price_volatility': 0.03
                }
            },
            'rl_agent': {
                'enabled': not args.disable_agent,
                'model_path': args.model_path,
                'config': {
                    'decision_interval': 5.0 if args.fast_mode else 8.0,
                    'confidence_threshold': 0.2,
                    'max_price_change': 0.15
                }
            },
            'pricing_environment': {
                'enabled': not args.disable_environment,
                'config': {
                    'update_interval': 3.0 if args.fast_mode else 5.0,
                    'auto_advance': True
                }
            },
            'monitor': {
                'enabled': not args.disable_monitor
            }
        },
        'health_check_interval': 15.0 if args.fast_mode else 30.0,
        'startup_delay': 1.0 if args.fast_mode else 2.0
    }
    
    # Create and run orchestrator
    orchestrator = SystemOrchestrator(config)
    
    try:
        # Start system
        if not orchestrator.start_system():
            logger.error("Failed to start system")
            sys.exit(1)
        
        # Run system
        orchestrator.run_system(args.duration)
        
    except Exception as e:
        logger.error(f"System error: {e}")
    
    finally:
        # Stop system and get final stats
        final_results = orchestrator.stop_system()
        
        # Print final summary
        print("\n" + "="*80)
        print("FINAL SYSTEM SUMMARY")
        print("="*80)
        
        system_summary = orchestrator.get_system_summary()
        
        # Print key metrics
        runtime = system_summary['system_status']['runtime_seconds']
        print(f"Total Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
        print(f"Components Started: {len(system_summary['system_stats']['components_started'])}")
        print(f"Total Errors: {system_summary['system_status']['errors_count']}")
        
        # Print component performance
        for component, stats in system_summary['component_performance'].items():
            print(f"\n{component.upper()}:")
            if component == 'market_streamer':
                print(f"  Events Sent: {stats.get('total_events_sent', 0)}")
                print(f"  Events/sec: {stats.get('events_per_second', 0):.2f}")
            elif component == 'rl_agent':
                print(f"  Pricing Actions: {stats.get('total_pricing_actions', 0)}")
                print(f"  Market Events Received: {stats.get('total_market_events', 0)}")
                print(f"  Model Predictions: {stats.get('model_predictions', 0)}")
            elif component == 'pricing_environment':
                print(f"  Market Updates: {stats.get('total_market_updates', 0)}")
                print(f"  Environment Steps: {stats.get('environment_steps', 0)}")
                print(f"  Total Profit: ${stats.get('total_profit', 0):.2f}")
        
        # Save detailed results
        with open(f"system_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump({
                'final_results': final_results,
                'system_summary': system_summary
            }, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to system_results_*.json")
        print("="*80)


if __name__ == "__main__":
    main()