# Instructions for Using the Algorithmic Trading Strategy

This document provides instructions on how to use our algorithmic trading strategy for the Amherst Quant Competition 2025.

## File Structure

- `allocate.py`: Main submission file containing our optimized strategy implementation
- `backtest.py`: Script to locally test the strategy on the training data
- `requirements.txt`: Dependencies required to run the code
- `README.md`: Overview of the strategy approach
- `STRATEGY.md`: Detailed explanation of the strategies and implementation

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run a local backtest to evaluate the strategy:
   ```
   python backtest.py
   ```
   This will show the strategy's performance metrics and save a performance plot.

## Submitting to the Competition

For the competition, you should submit the `allocate.py` file. This file contains an optimized version of the strategy with parameters tuned for the competition metrics.

To prepare your submission:

1. Ensure the class name is `PortfolioStrategy` and the main allocation function is named `allocate`
2. Submit the file through Gradescope before the deadline
3. No modifications are needed as the file is already named correctly

## Strategy Details

For a detailed explanation of the strategies implemented in our algorithm, please refer to the `STRATEGY.md` file, which includes:

- Detailed explanations of each component strategy
- Implementation details with code snippets
- Risk management techniques
- Performance optimization approaches
- Robustness features

## Parameters Tuning

The strategy contains several parameters that can be tuned to optimize performance:

- **Strategy Weights**: The relative importance of each sub-strategy (momentum, volatility, etc.)
- **Lookback Windows**: Time windows for various calculations
- **Position Constraints**: Maximum and minimum position sizes
- **Risk Management Parameters**: Volatility targeting and market regime detection thresholds

To tune these parameters, modify them in the `__init__` method of the `PortfolioStrategy` class and run backtests to see how they affect performance.

## Extending the Strategy

The modular nature of our strategy makes it easy to extend:

1. Add new sub-strategies by creating additional calculation methods
2. Adjust the weighting scheme to incorporate your new strategies
3. Modify the risk management approach to suit different market conditions

## Troubleshooting

- If you encounter numerical issues, check the handling of zero or NaN values
- For performance issues, consider reducing the complexity of calculations
- If backtesting shows unexpected results, verify your data handling and return calculations

## Competition Metrics Focus

Our strategy is designed to perform well across all evaluation metrics:

- **Sharpe Ratio**: Through effective diversification and risk parity allocation
- **Sortino Ratio**: By adaptive volatility targeting and market regime detection
- **Max Drawdown**: Through dynamic risk allocation during adverse market conditions
- **Volatility**: Via minimum volatility component and volatility scaling
- **Annual Return**: Momentum and trend following components capture upside
- **Calmar Ratio**: Combination of drawdown reduction and return enhancement
- **Win Rate**: Moving average and trend following systems improve directionality

Good luck in the competition! 