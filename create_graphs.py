"""
Simple script to create graphs from CSV files
Run this after your kafka system finishes
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def create_graphs_from_csv():
    # Find the most recent CSV files
    log_dir = "performance_logs"
    
    if not os.path.exists(log_dir):
        print(f"Directory {log_dir} not found")
        return
    
    # Get most recent pricing actions file
    pricing_files = [f for f in os.listdir(log_dir) if f.startswith('pricing_actions_')]
    if not pricing_files:
        print("No pricing actions CSV files found")
        return
    
    latest_pricing_file = os.path.join(log_dir, max(pricing_files))
    print(f"Using pricing file: {latest_pricing_file}")
    
    # Load data
    try:
        df = pd.read_csv(latest_pricing_file)
        print(f"Loaded {len(df)} pricing actions")
        
        if len(df) < 2:
            print("Not enough data to create graphs")
            return
        
        # Create graphs
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        fig.suptitle('RL Agent Performance Analysis', fontsize=16)
        
        # 1. Price Evolution
        axes[0].plot(range(len(df)), df['price'], 'b-', linewidth=2, marker='o')
        axes[0].set_title('Price Evolution Over Actions')
        axes[0].set_ylabel('Price ($)')
        axes[0].set_xlabel('Action Number')
        axes[0].grid(True, alpha=0.3)
        
        
        # 2. Decision Types
        decision_counts = df['decision_type'].value_counts()
        axes[1].pie(decision_counts.values, labels=decision_counts.index, 
                     autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Decision Types')
        
        plt.tight_layout()
        
        # Save graph
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        graph_file = os.path.join(log_dir, f'rl_performance_graphs_{timestamp}.png')
        plt.savefig(graph_file, dpi=300, bbox_inches='tight')
        print(f"Graphs saved to: {graph_file}")
        
        # Show statistics
        print("\nPerformance Statistics:")
        print(f"Total Actions: {len(df)}")
        print(f"Average Price: ${df['price'].mean():.2f}")
        print(f"Price Range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"Average Confidence: {df['confidence'].mean():.3f}")
        
        decision_breakdown = df['decision_type'].value_counts()
        print(f"\nDecision Breakdown:")
        for decision_type, count in decision_breakdown.items():
            print(f"  {decision_type}: {count} ({count/len(df)*100:.1f}%)")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating graphs: {e}")

if __name__ == "__main__":
    create_graphs_from_csv()