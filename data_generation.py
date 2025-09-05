import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import random
from dataclasses import dataclass
from enum import Enum


@dataclass
class MarketEvent:
    """Structure for market events"""
    timestamp: datetime
    event_type: str
    impact: float
    duration_hours: int
    description: str


class CustomerSegment(Enum):
    """Different types of customers"""
    PRICE_SENSITIVE = "price_sensitive"    # Students, budget shoppers
    BRAND_LOYAL = "brand_loyal"           # Premium customers
    IMPULSE_BUYERS = "impulse_buyers"     # Convenience shoppers
    BULK_BUYERS = "bulk_buyers"           # Business customers


class CompetitorStrategy(Enum):
    """Different competitor pricing strategies"""
    COPYCAT = "copycat"          # Follows our prices
    AGGRESSIVE = "aggressive"    # Always tries to undercut
    PREMIUM = "premium"         # Maintains high prices
    RANDOM = "random"           # Unpredictable pricing


class CustomerBehaviorGenerator:
    """Generates realistic customer demand patterns"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'base_demand': 50.0,
            'max_demand': 200.0,
            'segments': {
                CustomerSegment.PRICE_SENSITIVE: {'ratio': 0.4, 'elasticity': -2.5},
                CustomerSegment.BRAND_LOYAL: {'ratio': 0.3, 'elasticity': -0.8},
                CustomerSegment.IMPULSE_BUYERS: {'ratio': 0.2, 'elasticity': -1.5},
                CustomerSegment.BULK_BUYERS: {'ratio': 0.1, 'elasticity': -1.2}
            }
        }
    
    def generate_demand(self, 
                       price: float, 
                       base_price: float,
                       time_step: int,
                       external_factors: Dict = None) -> float:
        """Generate demand based on price and market conditions"""
        
        external_factors = external_factors or {}
        total_demand = 0.0
        
        # Generate demand for each customer segment
        for segment, params in self.config['segments'].items():
            segment_demand = self._calculate_segment_demand(
                segment, params, price, base_price, time_step, external_factors
            )
            total_demand += segment_demand
        
        # Add time-based patterns
        total_demand *= self._get_time_multiplier(time_step)
        
        # Add random noise
        noise = np.random.normal(1.0, 0.15)
        total_demand *= max(0.1, noise)  # Ensure demand doesn't go negative
        
        return min(total_demand, self.config['max_demand'])
    
    def _calculate_segment_demand(self, 
                                segment: CustomerSegment, 
                                params: Dict,
                                price: float,
                                base_price: float,
                                time_step: int,
                                external_factors: Dict) -> float:
        """Calculate demand for a specific customer segment"""
        
        # Base demand for this segment
        segment_base_demand = self.config['base_demand'] * params['ratio']
        
        # Price elasticity effect
        price_ratio = price / base_price
        price_effect = price_ratio ** params['elasticity']
        
        # Segment-specific time patterns
        if segment == CustomerSegment.IMPULSE_BUYERS:
            # More active during lunch and evening
            hour = time_step % 24
            time_boost = 1.2 if hour in [12, 13, 18, 19, 20] else 1.0
        elif segment == CustomerSegment.BULK_BUYERS:
            # More active on weekdays
            day_of_week = (time_step // 24) % 7
            time_boost = 1.3 if day_of_week < 5 else 0.7
        else:
            time_boost = 1.0
        
        # External factor effects
        event_multiplier = external_factors.get('demand_multiplier', 1.0)
        
        return segment_base_demand * price_effect * time_boost * event_multiplier
    
    def _get_time_multiplier(self, time_step: int) -> float:
        """Get time-based demand multiplier"""
        
        # Hour of day effect (24-hour cycle)
        hour = time_step % 24
        hourly_pattern = 0.7 + 0.6 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 6 PM
        
        # Day of week effect (7-day cycle)
        day_of_week = (time_step // 24) % 7
        if day_of_week in [5, 6]:  # Weekend
            weekly_pattern = 1.3
        else:  # Weekday
            weekly_pattern = 1.0
        
        return hourly_pattern * weekly_pattern


class CompetitorSimulator:
    """Simulates competitor pricing behavior"""
    
    def __init__(self, n_competitors: int = 3):
        self.competitors = []
        
        # Create competitors with different strategies
        strategies = [CompetitorStrategy.COPYCAT, CompetitorStrategy.AGGRESSIVE, CompetitorStrategy.PREMIUM]
        
        for i in range(n_competitors):
            competitor = {
                'id': i,
                'strategy': strategies[i % len(strategies)],
                'base_price': 95.0 + np.random.normal(0, 10),
                'current_price': 95.0 + np.random.normal(0, 10),
                'reactivity': 0.1 + np.random.random() * 0.4,  # How fast they react
                'last_price_change': 0
            }
            self.competitors.append(competitor)
    
    def update_competitor_prices(self, our_price: float, time_step: int) -> List[float]:
        """Update competitor prices based on their strategies"""
        
        for competitor in self.competitors:
            strategy = competitor['strategy']
            
            if strategy == CompetitorStrategy.COPYCAT:
                # Try to match our price but slightly lower
                target_price = our_price * 0.98
                adjustment = competitor['reactivity'] * (target_price - competitor['current_price'])
                competitor['current_price'] += adjustment
                
            elif strategy == CompetitorStrategy.AGGRESSIVE:
                # Always try to be cheapest in market
                market_min = min(our_price, min(c['current_price'] for c in self.competitors))
                target_price = market_min * 0.95
                adjustment = competitor['reactivity'] * (target_price - competitor['current_price'])
                competitor['current_price'] += adjustment
                
            elif strategy == CompetitorStrategy.PREMIUM:
                # Maintain premium positioning
                target_price = max(competitor['base_price'] * 1.1, our_price * 1.05)
                adjustment = competitor['reactivity'] * (target_price - competitor['current_price'])
                competitor['current_price'] += adjustment
                
            elif strategy == CompetitorStrategy.RANDOM:
                # Random walk pricing
                change = np.random.normal(0, 2)
                competitor['current_price'] += change
            
            # Keep prices reasonable
            competitor['current_price'] = max(60, min(200, competitor['current_price']))
        
        return [c['current_price'] for c in self.competitors]
    
    def get_current_prices(self) -> List[float]:
        """Get current competitor prices"""
        return [c['current_price'] for c in self.competitors]


class ExternalEventGenerator:
    """Generates external market events (holidays, promotions, economic changes)"""
    
    def __init__(self):
        self.active_events = []
        self.event_templates = [
            {'type': 'holiday_sale', 'demand_multiplier': 2.0, 'duration': 72, 'probability': 0.02},
            {'type': 'competitor_promotion', 'demand_multiplier': 0.7, 'duration': 48, 'probability': 0.03},
            {'type': 'supply_shortage', 'demand_multiplier': 1.5, 'duration': 120, 'probability': 0.01},
            {'type': 'economic_downturn', 'demand_multiplier': 0.6, 'duration': 240, 'probability': 0.005},
            {'type': 'viral_trend', 'demand_multiplier': 3.0, 'duration': 24, 'probability': 0.01},
            {'type': 'seasonal_peak', 'demand_multiplier': 1.8, 'duration': 168, 'probability': 0.015}
        ]
    
    def generate_events(self, time_step: int) -> List[MarketEvent]:
        """Generate new external events"""
        new_events = []
        
        # Remove expired events
        self.active_events = [e for e in self.active_events 
                            if (time_step - e.start_time) < e.duration_hours]
        
        # Generate new events based on probability
        for template in self.event_templates:
            if np.random.random() < template['probability']:
                event = MarketEvent(
                    timestamp=datetime.now() + timedelta(hours=time_step),
                    event_type=template['type'],
                    impact=template['demand_multiplier'],
                    duration_hours=template['duration'],
                    description=f"Market event: {template['type']}"
                )
                event.start_time = time_step
                self.active_events.append(event)
                new_events.append(event)
        
        return new_events
    
    def get_current_impact(self, time_step: int) -> Dict[str, float]:
        """Get current impact of all active events"""
        impact = {'demand_multiplier': 1.0}
        
        for event in self.active_events:
            # Decay effect over time
            elapsed = time_step - event.start_time
            decay_factor = max(0.1, 1.0 - (elapsed / event.duration_hours))
            
            impact['demand_multiplier'] *= (1.0 + (event.impact - 1.0) * decay_factor)
        
        return impact


class MarketDataGenerator:
    """Main class that orchestrates all data generation"""
    
    def __init__(self, product_config: Dict = None):
        self.product_config = product_config or {
            'name': 'Premium Laptop',
            'category': 'Electronics',
            'base_price': 1000.0,
            'cost': 700.0,
            'initial_inventory': 500
        }
        
        # Initialize components
        self.customer_generator = CustomerBehaviorGenerator()
        self.competitor_simulator = CompetitorSimulator(n_competitors=4)
        self.event_generator = ExternalEventGenerator()
        
        # Market state
        self.current_inventory = self.product_config['initial_inventory']
        self.time_step = 0
        
        # Data storage
        self.market_history = []
    
    def generate_timestep_data(self, our_price: float) -> Dict[str, any]:
        """Generate complete market data for one time step"""
        
        # Generate external events
        new_events = self.event_generator.generate_events(self.time_step)
        current_impact = self.event_generator.get_current_impact(self.time_step)
        
        # Update competitor prices
        competitor_prices = self.competitor_simulator.update_competitor_prices(
            our_price, self.time_step
        )
        
        # Generate customer demand
        demand = self.customer_generator.generate_demand(
            price=our_price,
            base_price=self.product_config['base_price'],
            time_step=self.time_step,
            external_factors=current_impact
        )
        
        # Calculate sales (limited by inventory)
        sales = min(demand, self.current_inventory)
        
        # Update inventory
        self.current_inventory = max(0, self.current_inventory - sales)
        
        # Restock simulation (supplier delivers every few days)
        if self.time_step % 48 == 0:  # Every 2 days
            restock_amount = min(200, self.product_config['initial_inventory'] - self.current_inventory)
            self.current_inventory += restock_amount
        
        # Create market data point
        market_data = {
            'timestamp': datetime.now() + timedelta(hours=self.time_step),
            'time_step': self.time_step,
            'our_price': our_price,
            'competitor_prices': competitor_prices.copy(),
            'demand': demand,
            'sales': sales,
            'inventory': self.current_inventory,
            'external_events': [e.event_type for e in new_events],
            'demand_multiplier': current_impact['demand_multiplier'],
            'hour_of_day': self.time_step % 24,
            'day_of_week': (self.time_step // 24) % 7,
            'revenue': our_price * sales,
            'profit': (our_price - self.product_config['cost']) * sales
        }
        
        # Store in history
        self.market_history.append(market_data)
        
        # Increment time
        self.time_step += 1
        
        return market_data
    
    def generate_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Generate historical market data for initial training"""
        
        print(f"Generating {days} days of historical market data...")
        historical_data = []
        
        # Reset state
        self.time_step = 0
        self.current_inventory = self.product_config['initial_inventory']
        
        total_timesteps = days * 24  # Hourly data
        
        for step in range(total_timesteps):
            # Simulate different pricing strategies over time
            if step < total_timesteps * 0.3:
                # Early period: stable pricing
                our_price = self.product_config['base_price'] + np.random.normal(0, 5)
            elif step < total_timesteps * 0.6:
                # Middle period: some experimentation
                our_price = self.product_config['base_price'] * (0.9 + 0.3 * np.random.random())
            else:
                # Later period: more aggressive pricing
                our_price = self.product_config['base_price'] * (0.8 + 0.5 * np.random.random())
            
            # Generate market data for this timestep
            market_data = self.generate_timestep_data(our_price)
            historical_data.append(market_data)
            
            if step % (total_timesteps // 10) == 0:
                print(f"Progress: {step/total_timesteps*100:.1f}%")
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Add derived features
        df['competitor_avg'] = df['competitor_prices'].apply(lambda x: np.mean(x))
        df['competitor_min'] = df['competitor_prices'].apply(lambda x: np.min(x))
        df['price_vs_competitors'] = df['our_price'] / df['competitor_avg']
        df['profit_margin'] = (df['our_price'] - self.product_config['cost']) / df['our_price']
        
        print(f"Generated {len(df)} data points!")
        return df
    
    def get_market_statistics(self) -> Dict[str, any]:
        """Get summary statistics of generated market data"""
        if not self.market_history:
            return {}
        
        df = pd.DataFrame(self.market_history)
        
        stats = {
            'avg_demand': df['demand'].mean(),
            'avg_sales': df['sales'].mean(),
            'total_revenue': df['revenue'].sum(),
            'total_profit': df['profit'].sum(),
            'avg_price': df['our_price'].mean(),
            'price_volatility': df['our_price'].std(),
            'avg_competitor_price': df['competitor_prices'].apply(lambda x: np.mean(x)).mean(),
            'demand_peak_hour': df.groupby('hour_of_day')['demand'].mean().idxmax(),
            'most_profitable_day': df.groupby('day_of_week')['profit'].sum().idxmax()
        }
        
        return stats


class RealTimeMarketSimulator:
    """Simulates real-time market data generation"""
    
    def __init__(self, product_config: Dict = None):
        self.data_generator = MarketDataGenerator(product_config)
        self.is_running = False
        
    def start_simulation(self, our_price: float = None):
        """Start real-time market simulation"""
        our_price = our_price or self.data_generator.product_config['base_price']
        self.is_running = True
        print("Starting real-time market simulation...")
        
        return self.data_generator.generate_timestep_data(our_price)
    
    def step_simulation(self, our_price: float) -> Dict[str, any]:
        """Generate one step of market data"""
        if not self.is_running:
            raise ValueError("Simulation not started. Call start_simulation() first.")
        
        return self.data_generator.generate_timestep_data(our_price)
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        print("Market simulation stopped.")
        return self.data_generator.get_market_statistics()


# Utility functions for data analysis and visualization
def analyze_price_elasticity(data: pd.DataFrame) -> Dict[str, float]:
    """Analyze price elasticity from generated data"""
    
    # Calculate correlation between price and demand
    price_demand_corr = data['our_price'].corr(data['demand'])
    
    # Calculate elasticity using log-log regression
    log_price = np.log(data['our_price'])
    log_demand = np.log(data['demand'] + 1)  # Add 1 to avoid log(0)
    
    # Simple linear regression
    coef = np.polyfit(log_price, log_demand, 1)[0]
    
    return {
        'price_demand_correlation': price_demand_corr,
        'estimated_elasticity': coef,
        'avg_demand_at_base_price': data[data['our_price'].between(95, 105)]['demand'].mean(),
        'demand_volatility': data['demand'].std()
    }


def create_market_dataset(config: Dict = None, days: int = 30) -> Tuple[pd.DataFrame, Dict]:
    """Create a complete market dataset for training"""
    
    print("Creating market dataset...")
    
    # Initialize generator
    generator = MarketDataGenerator(config)
    
    # Generate historical data
    data = generator.generate_historical_data(days)
    
    # Analyze the generated data
    stats = generator.get_market_statistics()
    elasticity_analysis = analyze_price_elasticity(data)
    
    analysis = {
        'market_stats': stats,
        'elasticity_analysis': elasticity_analysis,
        'data_shape': data.shape,
        'time_range': {
            'start': data['timestamp'].min(),
            'end': data['timestamp'].max()
        }
    }
    
    return data, analysis


# Example usage and testing
if __name__ == "__main__":
    # Configuration for our product
    laptop_config = {
        'name': 'Gaming Laptop',
        'base_price': 1200.0,
        'cost': 800.0,
        'initial_inventory': 1000
    }
    
    # Generate historical dataset
    print("Generating market dataset...")
    market_data, analysis = create_market_dataset(laptop_config, days=14)
    
    # Display analysis
    print("\nMarket Analysis:")
    print(f"Total data points: {len(market_data)}")
    print(f"Average demand: {analysis['market_stats']['avg_demand']:.1f}")
    print(f"Average sales: {analysis['market_stats']['avg_sales']:.1f}")
    print(f"Total profit: ${analysis['market_stats']['total_profit']:.2f}")
    print(f"Price elasticity: {analysis['elasticity_analysis']['estimated_elasticity']:.2f}")
    
    # Save dataset
    market_data.to_csv('market_data.csv', index=False)
    
    with open('market_analysis.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, dict):
                json_analysis[key] = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                    for k, v in value.items() if not isinstance(v, datetime)}
            else:
                json_analysis[key] = value
        json.dump(json_analysis, f, indent=2, default=str)
    
    print("\nDataset saved to 'market_data.csv'")
    print("Analysis saved to 'market_analysis.json'")
    
    # Test real-time simulation
    print("\nTesting real-time simulation...")
    simulator = RealTimeMarketSimulator(laptop_config)
    
    # Simulate 10 steps
    simulator.start_simulation(1200.0)
    for i in range(10):
        price = 1200 + np.random.normal(0, 50)  # Random price changes
        data_point = simulator.step_simulation(price)
        print(f"Step {i+1}: Price=${price:.2f}, Demand={data_point['demand']:.1f}, Sales={data_point['sales']:.1f}")
    
    final_stats = simulator.stop_simulation()
    print(f"\nSimulation complete. Final profit: ${final_stats['total_profit']:.2f}")