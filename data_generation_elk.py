# market_data_generators_elk.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any
import json
import random
from dataclasses import dataclass
from enum import Enum
from pricing_env_elk import DynamicPricingEnv, ELKLogger

# Import ELKLogger from your ELK-enabled environment (existing file)
# Make sure 'pricing_env_elk.py' is in same directory or PYTHONPATH
from pricing_env_elk import ELKLogger

@dataclass
class MarketEvent:
    """Structure for market events"""
    timestamp: datetime
    event_type: str
    impact: float
    duration_hours: int
    description: str
    start_time: int = 0

class CustomerSegment(Enum):
    PRICE_SENSITIVE = "price_sensitive"
    BRAND_LOYAL = "brand_loyal"
    IMPULSE_BUYERS = "impulse_buyers"
    BULK_BUYERS = "bulk_buyers"

class CompetitorStrategy(Enum):
    COPYCAT = "copycat"
    AGGRESSIVE = "aggressive"
    PREMIUM = "premium"
    RANDOM = "random"

class CustomerBehaviorGenerator:
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
        external_factors = external_factors or {}
        total_demand = 0.0
        for segment, params in self.config['segments'].items():
            segment_demand = self._calculate_segment_demand(
                segment, params, price, base_price, time_step, external_factors
            )
            total_demand += segment_demand
        total_demand *= self._get_time_multiplier(time_step)
        noise = np.random.normal(1.0, 0.15)
        total_demand *= max(0.1, float(noise))
        return min(float(total_demand), float(self.config['max_demand']))
    
    def _calculate_segment_demand(self, 
                                segment: CustomerSegment, 
                                params: Dict,
                                price: float,
                                base_price: float,
                                time_step: int,
                                external_factors: Dict) -> float:
        segment_base_demand = float(self.config['base_demand'] * params['ratio'])
        price_ratio = float(price / base_price)
        price_effect = price_ratio ** params['elasticity']
        if segment == CustomerSegment.IMPULSE_BUYERS:
            hour = time_step % 24
            time_boost = 1.2 if hour in [12, 13, 18, 19, 20] else 1.0
        elif segment == CustomerSegment.BULK_BUYERS:
            day_of_week = (time_step // 24) % 7
            time_boost = 1.3 if day_of_week < 5 else 0.7
        else:
            time_boost = 1.0
        event_multiplier = float(external_factors.get('demand_multiplier', 1.0))
        return float(segment_base_demand * price_effect * time_boost * event_multiplier)
    
    def _get_time_multiplier(self, time_step: int) -> float:
        hour = time_step % 24
        hourly_pattern = 0.7 + 0.6 * np.sin(2 * np.pi * (hour - 6) / 24)
        day_of_week = (time_step // 24) % 7
        weekly_pattern = 1.3 if day_of_week in [5, 6] else 1.0
        return float(hourly_pattern * weekly_pattern)


class CompetitorSimulator:
    def __init__(self, n_competitors: int = 3):
        self.competitors = []
        strategies = [CompetitorStrategy.COPYCAT, CompetitorStrategy.AGGRESSIVE, CompetitorStrategy.PREMIUM, CompetitorStrategy.RANDOM]
        for i in range(n_competitors):
            competitor = {
                'id': i,
                'strategy': strategies[i % len(strategies)],
                'base_price': float(95.0 + np.random.normal(0, 10)),
                'current_price': float(95.0 + np.random.normal(0, 10)),
                'reactivity': float(0.1 + np.random.random() * 0.4),
                'last_price_change': 0
            }
            self.competitors.append(competitor)
    
    def update_competitor_prices(self, our_price: float, time_step: int) -> List[float]:
        for competitor in self.competitors:
            strategy = competitor['strategy']
            if strategy == CompetitorStrategy.COPYCAT:
                target_price = our_price * 0.98
                adjustment = competitor['reactivity'] * (target_price - competitor['current_price'])
                competitor['current_price'] += adjustment
            elif strategy == CompetitorStrategy.AGGRESSIVE:
                market_min = min(our_price, min(c['current_price'] for c in self.competitors))
                target_price = market_min * 0.95
                adjustment = competitor['reactivity'] * (target_price - competitor['current_price'])
                competitor['current_price'] += adjustment
            elif strategy == CompetitorStrategy.PREMIUM:
                target_price = max(competitor['base_price'] * 1.1, our_price * 1.05)
                adjustment = competitor['reactivity'] * (target_price - competitor['current_price'])
                competitor['current_price'] += adjustment
            else:
                competitor['current_price'] += float(np.random.normal(0, 2))
            competitor['current_price'] = float(max(60, min(200, competitor['current_price'])))
        return [float(c['current_price']) for c in self.competitors]
    
    def get_current_prices(self) -> List[float]:
        return [float(c['current_price']) for c in self.competitors]


class ExternalEventGenerator:
    def __init__(self):
        self.active_events: List[MarketEvent] = []
        self.event_templates = [
            {'type': 'holiday_sale', 'demand_multiplier': 2.0, 'duration': 72, 'probability': 0.02},
            {'type': 'competitor_promotion', 'demand_multiplier': 0.7, 'duration': 48, 'probability': 0.03},
            {'type': 'supply_shortage', 'demand_multiplier': 1.5, 'duration': 120, 'probability': 0.01},
            {'type': 'economic_downturn', 'demand_multiplier': 0.6, 'duration': 240, 'probability': 0.005},
            {'type': 'viral_trend', 'demand_multiplier': 3.0, 'duration': 24, 'probability': 0.01},
            {'type': 'seasonal_peak', 'demand_multiplier': 1.8, 'duration': 168, 'probability': 0.015}
        ]
    
    def generate_events(self, time_step: int) -> List[MarketEvent]:
        new_events = []
        # Remove expired events
        self.active_events = [e for e in self.active_events if (time_step - e.start_time) < e.duration_hours]
        for template in self.event_templates:
            if np.random.random() < template['probability']:
                event = MarketEvent(
                    timestamp=datetime.now(timezone.utc) + timedelta(hours=time_step),
                    event_type=template['type'],
                    impact=float(template['demand_multiplier']),
                    duration_hours=int(template['duration']),
                    description=f"Market event: {template['type']}",
                    start_time=time_step
                )
                self.active_events.append(event)
                new_events.append(event)
        return new_events
    
    def get_current_impact(self, time_step: int) -> Dict[str, float]:
        impact = {'demand_multiplier': 1.0}
        for event in self.active_events:
            elapsed = time_step - event.start_time
            decay_factor = max(0.1, 1.0 - (elapsed / event.duration_hours))
            impact['demand_multiplier'] *= float(1.0 + (event.impact - 1.0) * decay_factor)
        return {'demand_multiplier': float(impact['demand_multiplier'])}


class MarketDataGenerator:
    def __init__(self, product_config: Dict = None, elk_logger: ELKLogger = None):
        self.product_config = product_config or {
            'name': 'Premium Laptop',
            'category': 'Electronics',
            'base_price': 1000.0,
            'cost': 700.0,
            'initial_inventory': 500
        }
        self.customer_generator = CustomerBehaviorGenerator()
        self.competitor_simulator = CompetitorSimulator(n_competitors=4)
        self.event_generator = ExternalEventGenerator()
        self.current_inventory = int(self.product_config['initial_inventory'])
        self.time_step = 0
        self.market_history: List[Dict[str, Any]] = []
        # ELK logger (use provided or create a new one)
        self.elk_logger = elk_logger or ELKLogger()
    
    def _to_python_types(self, obj: Any) -> Any:
        """
        Helper to convert numpy scalars/arrays to native Python types for JSON/ELK.
        """
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return [self._to_python_types(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._to_python_types(v) for k, v in obj.items()}
        return obj

    def generate_timestep_data(self, our_price: float) -> Dict[str, Any]:
        new_events = self.event_generator.generate_events(self.time_step)
        current_impact = self.event_generator.get_current_impact(self.time_step)
        competitor_prices = self.competitor_simulator.update_competitor_prices(our_price, self.time_step)
        demand = self.customer_generator.generate_demand(
            price=our_price,
            base_price=self.product_config['base_price'],
            time_step=self.time_step,
            external_factors=current_impact
        )
        sales = float(min(demand, self.current_inventory))
        old_inventory = self.current_inventory
        self.current_inventory = int(max(0, self.current_inventory - sales))
        if self.time_step % 48 == 0 and self.time_step != 0:
            restock_amount = min(200, int(self.product_config['initial_inventory']) - self.current_inventory)
            self.current_inventory += restock_amount
        market_data = {
            'timestamp': (datetime.now(timezone.utc) + timedelta(hours=self.time_step)).isoformat(),
            'time_step': int(self.time_step),
            'product_name': self.product_config.get('name'),
            'our_price': float(our_price),
            'competitor_prices': [float(p) for p in competitor_prices],
            'demand': float(demand),
            'sales': float(sales),
            'inventory': int(self.current_inventory),
            'external_events': [e.event_type for e in new_events],
            'demand_multiplier': float(current_impact['demand_multiplier']),
            'hour_of_day': int(self.time_step % 24),
            'day_of_week': int((self.time_step // 24) % 7),
            'revenue': float(our_price * sales),
            'profit': float((our_price - self.product_config['cost']) * sales)
        }
        # Save and advance
        self.market_history.append(market_data)
        # ELK log for this timestep
        try:
            log_entry = self._to_python_types(market_data)
            # Use index name month-based for easier management
            self.elk_logger.log_environment_state({
                "event_subtype": "market_timestep",
                **log_entry
            }, index_name="market-data")
        except Exception as e:
            # swallow exception but log to ELK logger itself
            self.elk_logger.logger.error("elk_log_failed", error=str(e), step=self.time_step)
        self.time_step += 1
        return market_data
    
    def generate_historical_data(self, days: int = 30) -> pd.DataFrame:
        total_timesteps = int(days * 24)
        self.time_step = 0
        self.current_inventory = int(self.product_config['initial_inventory'])
        historical_data = []
        # Log generation start
        self.elk_logger.log_performance_metrics({
            "event_subtype": "generation_start",
            "product": self.product_config.get('name'),
            "days": int(days),
            "total_timesteps": total_timesteps
        }, index_name="market-generation")
        for step in range(total_timesteps):
            if step < total_timesteps * 0.3:
                our_price = float(self.product_config['base_price'] + np.random.normal(0, 5))
            elif step < total_timesteps * 0.6:
                our_price = float(self.product_config['base_price'] * (0.9 + 0.3 * np.random.random()))
            else:
                our_price = float(self.product_config['base_price'] * (0.8 + 0.5 * np.random.random()))
            market_data = self.generate_timestep_data(our_price)
            historical_data.append(market_data)
            if step % max(1, (total_timesteps // 10)) == 0:
                pct = float(step / max(1, total_timesteps) * 100.0)
                # Progress log to ELK
                self.elk_logger.log_performance_metrics({
                    "event_subtype": "generation_progress",
                    "step": int(step),
                    "percent_complete": float(pct),
                    "time_step": int(self.time_step)
                }, index_name="market-generation")
        df = pd.DataFrame(historical_data)
        if 'competitor_prices' in df.columns:
            df['competitor_avg'] = df['competitor_prices'].apply(lambda x: float(np.mean(x)) if len(x) else 0.0)
            df['competitor_min'] = df['competitor_prices'].apply(lambda x: float(np.min(x)) if len(x) else 0.0)
        df['price_vs_competitors'] = df['our_price'] / df['competitor_avg'].replace(0, np.nan)
        df['profit_margin'] = (df['our_price'] - self.product_config['cost']) / df['our_price']
        # Final generation stats and ELK log
        stats = self.get_market_statistics(from_df=df)
        self.elk_logger.log_performance_metrics({
            "event_subtype": "generation_complete",
            "product": self.product_config.get('name'),
            "days": int(days),
            "generated_points": int(len(df)),
            **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in stats.items()}
        }, index_name="market-generation")
        return df

    def get_market_statistics(self, from_df: pd.DataFrame = None) -> Dict[str, Any]:
        df = from_df if from_df is not None else pd.DataFrame(self.market_history)
        if df.empty:
            return {}
        stats = {
            'avg_demand': float(df['demand'].mean()),
            'avg_sales': float(df['sales'].mean()),
            'total_revenue': float(df['revenue'].sum()),
            'total_profit': float(df['profit'].sum()),
            'avg_price': float(df['our_price'].mean()),
            'price_volatility': float(df['our_price'].std()),
            'avg_competitor_price': float(df['competitor_prices'].apply(lambda x: np.mean(x)).mean()),
            'demand_peak_hour': int(df.groupby('hour_of_day')['demand'].mean().idxmax()),
            'most_profitable_day': int(df.groupby('day_of_week')['profit'].sum().idxmax())
        }
        return stats

# Helper create function
def create_market_dataset(config: Dict = None, days: int = 30, elk_logger: ELKLogger = None) -> Tuple[pd.DataFrame, Dict]:
    generator = MarketDataGenerator(config, elk_logger=elk_logger)
    data = generator.generate_historical_data(days)
    stats = generator.get_market_statistics(from_df=data)
    elasticity = {
        'price_demand_correlation': float(data['our_price'].corr(data['demand'])),
        'estimated_elasticity': float(np.polyfit(np.log(data['our_price']), np.log(data['demand'] + 1), 1)[0]) if len(data) > 5 else 0.0,
        'avg_demand_at_base_price': float(data[data['our_price'].between(generator.product_config['base_price']*0.95, generator.product_config['base_price']*1.05)]['demand'].mean()),
        'demand_volatility': float(data['demand'].std())
    }
    analysis = {
        'market_stats': stats,
        'elasticity_analysis': elasticity,
        'data_shape': tuple(data.shape),
        'time_range': {
            'start': str(data['timestamp'].min()) if not data.empty else None,
            'end': str(data['timestamp'].max()) if not data.empty else None
        }
    }
    # Log final analysis to ELK
    if generator.elk_logger:
        generator.elk_logger.log_performance_metrics({
            "event_subtype": "dataset_analysis",
            "product": generator.product_config.get('name'),
            **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in stats.items()},
            "elasticity_estimate": float(elasticity.get('estimated_elasticity', 0.0)),
            "rows": int(data.shape[0]),
            "cols": int(data.shape[1])
        }, index_name="market-analysis")
    return data, analysis

# Example usage
if __name__ == "__main__":
    elk = ELKLogger()
    laptop_config = {
        'name': 'Gaming Laptop',
        'base_price': 1200.0,
        'cost': 800.0,
        'initial_inventory': 1000
    }
    market_data, analysis = create_market_dataset(laptop_config, days=7, elk_logger=elk)
    market_data.to_csv('market_data_elk.csv', index=False)
    with open('market_analysis_elk.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print("Saved market_data_elk.csv and market_analysis_elk.json")
