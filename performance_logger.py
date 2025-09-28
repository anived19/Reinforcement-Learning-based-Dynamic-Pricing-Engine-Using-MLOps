"""
Enhanced Performance Data Logger for RL Agent
Logs detailed action data for analysis and visualization
"""
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any
import os

class PerformanceLogger:
    """Logs and analyzes RL agent performance data"""
    
    def __init__(self, log_dir: str = "performance_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Data storage
        self.pricing_actions = []
        self.market_events = []
        self.rewards = []
        self.system_metrics = []
        
        # File paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.actions_file = os.path.join(log_dir, f"pricing_actions_{timestamp}.csv")
        self.events_file = os.path.join(log_dir, f"market_events_{timestamp}.csv")
        self.metrics_file = os.path.join(log_dir, f"system_metrics_{timestamp}.json")
    
    def log_pricing_action(self, action_data: Dict[str, Any]):
        """Log a pricing action with timestamp"""
        enhanced_data = {
            'timestamp': datetime.now().isoformat(),
            'price': action_data.get('price', 0),
            'previous_price': action_data.get('market_context', {}).get('previous_price', 0),
            'price_change_percent': action_data.get('price_change_percent', 0),
            'confidence': action_data.get('confidence', 0),
            'decision_type': action_data.get('decision_type', 'unknown'),
            'demand': action_data.get('market_context', {}).get('demand', 0),
            'inventory': action_data.get('market_context', {}).get('inventory', 0),
            'competitor_avg': action_data.get('market_context', {}).get('competitor_avg', 0),
            'decision_time_ms': action_data.get('decision_time_ms', 0)
        }
        
        self.pricing_actions.append(enhanced_data)
        
        # Write to CSV immediately for real-time analysis
        self._write_to_csv(self.actions_file, enhanced_data)
    
    def log_market_event(self, market_data: Dict[str, Any]):
        """Log market event data"""
        enhanced_data = {
            'timestamp': datetime.now().isoformat(),
            'our_price': market_data.get('our_price', 0),
            'demand': market_data.get('demand', 0),
            'sales': market_data.get('sales', 0),
            'inventory': market_data.get('inventory', 0),
            'competitor_prices': json.dumps(market_data.get('competitor_prices', [])),
            'demand_multiplier': market_data.get('demand_multiplier', 1.0),
            'hour_of_day': market_data.get('hour_of_day', 0),
            'day_of_week': market_data.get('day_of_week', 0)
        }
        
        self.market_events.append(enhanced_data)
        self._write_to_csv(self.events_file, enhanced_data)
    
    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system-wide performance metrics"""
        timestamped_metrics = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.system_metrics.append(timestamped_metrics)
    
    def _write_to_csv(self, filename: str, data: Dict[str, Any]):
        """Write data to CSV file"""
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as csvfile:
            if data:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
    
    def save_final_summary(self):
        """Save comprehensive summary at end of run"""
        with open(self.metrics_file, 'w') as f:
            json.dump({
                'system_metrics': self.system_metrics,
                'summary_stats': self.calculate_summary_stats()
            }, f, indent=2)
    
    def calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate performance summary statistics"""
        if not self.pricing_actions:
            return {}
        
        prices = [action['price'] for action in self.pricing_actions]
        price_changes = [action['price_change_percent'] for action in self.pricing_actions]
        confidences = [action['confidence'] for action in self.pricing_actions]
        
        return {
            'total_actions': len(self.pricing_actions),
            'avg_price': np.mean(prices),
            'price_volatility': np.std(prices),
            'avg_price_change': np.mean(np.abs(price_changes)),
            'max_price_change': np.max(np.abs(price_changes)),
            'avg_confidence': np.mean(confidences),
            'price_range': [np.min(prices), np.max(prices)],
            'model_vs_fallback': {
                'rl_model': sum(1 for a in self.pricing_actions if a['decision_type'] == 'rl_model'),
                'fallback': sum(1 for a in self.pricing_actions if a['decision_type'] == 'fallback')
            }
        }
    
    def create_performance_graphs(self):
        """Generate performance visualization graphs"""
        if not self.pricing_actions:
            print("No pricing actions to visualize")
            return
        
        # Convert to DataFrame for easier plotting
        df_actions = pd.DataFrame(self.pricing_actions)
        df_actions['timestamp'] = pd.to_datetime(df_actions['timestamp'])
        
        # Create multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL Agent Performance Analysis', fontsize=16)
        
        # 1. Price Evolution Over Time
        axes[0,0].plot(df_actions['timestamp'], df_actions['price'], 'b-', linewidth=2)
        axes[0,0].set_title('Price Evolution Over Time')
        axes[0,0].set_ylabel('Price ($)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Price Changes Distribution
        axes[0,1].hist(df_actions['price_change_percent'], bins=20, alpha=0.7, color='green')
        axes[0,1].set_title('Distribution of Price Changes')
        axes[0,1].set_xlabel('Price Change (%)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Confidence vs Price Change
        axes[1,0].scatter(df_actions['confidence'], df_actions['price_change_percent'], 
                         alpha=0.6, c='red', s=50)
        axes[1,0].set_title('Confidence vs Price Change')
        axes[1,0].set_xlabel('Confidence')
        axes[1,0].set_ylabel('Price Change (%)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Decision Types
        decision_counts = df_actions['decision_type'].value_counts()
        axes[1,1].pie(decision_counts.values, labels=decision_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Decision Types Distribution')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = os.path.join(self.log_dir, 
                                   f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Performance graphs saved to: {plot_filename}")
        
        return plot_filename
    
    def create_advanced_analysis(self):
        """Create advanced performance analysis"""
        if len(self.pricing_actions) < 10:
            print("Not enough data for advanced analysis")
            return
        
        df_actions = pd.DataFrame(self.pricing_actions)
        df_actions['timestamp'] = pd.to_datetime(df_actions['timestamp'])
        
        # Create advanced analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced RL Agent Analysis', fontsize=16)
        
        # 1. Price vs Demand Relationship
        axes[0,0].scatter(df_actions['demand'], df_actions['price'], alpha=0.6, c='blue')
        axes[0,0].set_title('Price vs Demand')
        axes[0,0].set_xlabel('Demand')
        axes[0,0].set_ylabel('Price ($)')
        
        # 2. Confidence Over Time
        axes[0,1].plot(df_actions['timestamp'], df_actions['confidence'], 'g-', alpha=0.7)
        axes[0,1].set_title('Confidence Evolution')
        axes[0,1].set_ylabel('Confidence')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Decision Time Analysis
        axes[0,2].boxplot(df_actions['decision_time_ms'])
        axes[0,2].set_title('Decision Time Distribution')
        axes[0,2].set_ylabel('Time (ms)')
        
        # 4. Rolling Average Price
        df_actions['price_ma'] = df_actions['price'].rolling(window=5).mean()
        axes[1,0].plot(df_actions['timestamp'], df_actions['price'], alpha=0.5, label='Actual')
        axes[1,0].plot(df_actions['timestamp'], df_actions['price_ma'], 'r-', linewidth=2, label='5-period MA')
        axes[1,0].set_title('Price with Moving Average')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Inventory Impact
        axes[1,1].scatter(df_actions['inventory'], df_actions['price_change_percent'], 
                         alpha=0.6, c='orange')
        axes[1,1].set_title('Inventory vs Price Change')
        axes[1,1].set_xlabel('Inventory')
        axes[1,1].set_ylabel('Price Change (%)')
        
        # 6. Performance Heatmap
        price_change_bins = pd.cut(df_actions['price_change_percent'], bins=5)
        confidence_bins = pd.cut(df_actions['confidence'], bins=5)
        heatmap_data = pd.crosstab(price_change_bins, confidence_bins)
        
        im = axes[1,2].imshow(heatmap_data.values, cmap='Blues', aspect='auto')
        axes[1,2].set_title('Price Change vs Confidence Heatmap')
        axes[1,2].set_xlabel('Confidence Bins')
        axes[1,2].set_ylabel('Price Change Bins')
        
        plt.tight_layout()
        
        # Save advanced analysis
        advanced_filename = os.path.join(self.log_dir, 
                                       f"advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(advanced_filename, dpi=300, bbox_inches='tight')
        print(f"Advanced analysis saved to: {advanced_filename}")
        
        return advanced_filename
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        stats = self.calculate_summary_stats()
        
        if not stats:
            return "No data available for report generation"
        
        report = f"""
=== RL AGENT PERFORMANCE REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS:
- Total Pricing Actions: {stats['total_actions']}
- Average Price: ${stats['avg_price']:.2f}
- Price Volatility (Std Dev): ${stats['price_volatility']:.2f}
- Average Price Change: {stats['avg_price_change']*100:.2f}%
- Maximum Price Change: {stats['max_price_change']*100:.2f}%
- Average Confidence: {stats['avg_confidence']:.3f}
- Price Range: ${stats['price_range'][0]:.2f} - ${stats['price_range'][1]:.2f}

DECISION BREAKDOWN:
- RL Model Decisions: {stats['model_vs_fallback']['rl_model']}
- Fallback Decisions: {stats['model_vs_fallback']['fallback']}
- Model Usage Rate: {stats['model_vs_fallback']['rl_model']/stats['total_actions']*100:.1f}%

FILES GENERATED:
- Pricing Actions: {self.actions_file}
- Market Events: {self.events_file}
- System Metrics: {self.metrics_file}
"""
        
        # Save report to file
        report_filename = os.path.join(self.log_dir, 
                                     f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(report)
        return report_filename