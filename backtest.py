import numpy as np
import pandas as pd
import json
from allocate import PortfolioStrategy
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load the training data from a JSON file
    """
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Get all the dates in chronological order
    dates = sorted(data['open'].keys(), reverse=True)
    
    print(f"Data loaded with {len(dates)} days of market data.")
    return data, dates

def backtest_strategy(data, dates):
    """
    Backtest the strategy on the provided data
    """
    print("Initializing strategy...")
    strategy = PortfolioStrategy()
    
    # Initialize tracking variables
    portfolio_values = [1.0]  # Start with $1
    allocations_history = []
    
    print("Running backtest...")
    for i, date in enumerate(dates):
        if i > 0:  # Skip the first day (we need returns data)
            # Get market data for the current day
            market_data = {
                'open': np.array([data['open'][date][stock] for stock in sorted(data['open'][date].keys())]),
                'close': np.array([data['close'][date][stock] for stock in sorted(data['close'][date].keys())]),
                'high': np.array([data['high'][date][stock] for stock in sorted(data['high'][date].keys())]),
                'low': np.array([data['low'][date][stock] for stock in sorted(data['low'][date].keys())]),
                'volume': np.array([data['volume'][date][stock] for stock in sorted(data['volume'][date].keys())])
            }
            
            # Get allocations from the strategy
            allocations = strategy.allocate(market_data)
            allocations_history.append(allocations)
            
            # Calculate returns
            prev_close = np.array([data['close'][dates[i-1]][stock] for stock in sorted(data['close'][dates[i-1]].keys())])
            curr_close = np.array([data['close'][date][stock] for stock in sorted(data['close'][date].keys())])
            
            # Calculate asset returns
            asset_returns = (curr_close - prev_close) / prev_close
            
            # Calculate portfolio return
            if i > 1:  # We have allocations from previous day
                portfolio_return = np.sum(allocations_history[-2] * asset_returns)
                portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
    
    print(f"Backtest completed over {len(dates)-1} trading days.")
    return portfolio_values, allocations_history

def calculate_metrics(portfolio_values):
    """
    Calculate performance metrics
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Annual return
    annual_return = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(returns)) - 1
    
    # Volatility
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    # Sortino Ratio
    negative_returns = returns[returns < 0]
    sortino = np.mean(returns) / np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else np.inf
    
    # Max Drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = abs(min(drawdowns))
    
    # Calmar Ratio
    calmar = annual_return / max_drawdown if max_drawdown > 0 else np.inf
    
    # Win Rate
    win_rate = len(returns[returns > 0]) / len(returns)
    
    metrics = {
        "Annual Return": annual_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
        "Win Rate": win_rate
    }
    
    return metrics

def plot_performance(portfolio_values, dates):
    """
    Plot portfolio performance
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title('Portfolio Performance')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    print("Performance plot saved to portfolio_performance.png")

def print_metrics(metrics):
    """
    Print performance metrics in a formatted way
    """
    print("\n===== Performance Metrics =====")
    print(f"Annual Return: {metrics['Annual Return']:.2%}")
    print(f"Volatility: {metrics['Volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['Sortino Ratio']:.2f}")
    print(f"Max Drawdown: {metrics['Max Drawdown']:.2%}")
    print(f"Calmar Ratio: {metrics['Calmar Ratio']:.2f}")
    print(f"Win Rate: {metrics['Win Rate']:.2%}")
    print("===============================")

def main():
    """
    Main function to run the backtest
    """
    # Load data
    data_path = "AlgorithmicTrading_Case_Materials_ACQC2025 2/training.json"
    data, dates = load_data(data_path)
    
    # Run backtest
    portfolio_values, allocations_history = backtest_strategy(data, dates)
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_values)
    
    # Print metrics
    print_metrics(metrics)
    
    # Plot performance
    try:
        plot_performance(portfolio_values, dates)
    except ImportError:
        print("Matplotlib not available for plotting.")

if __name__ == "__main__":
    main() 