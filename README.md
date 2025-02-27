# 2025-Amherst-Quant-Competition
1st ever Amherst Quant Competition Powered by Jane Street

# Algorithmic Trading Strategy for Amherst Quant Competition 2025

This repository contains an implementation of a multi-factor portfolio allocation strategy for the Amherst Quant Competition 2025.

## Strategy Overview

Our approach leverages a combination of well-established quantitative investment strategies to build a robust portfolio allocation model:

1. **Momentum Strategy (20%)**: Captures price momentum across multiple time horizons (10, 20, and 30 days) with emphasis on recent performance.

2. **Minimum Volatility Strategy (15%)**: Allocates more capital to assets with lower historical volatility to improve risk-adjusted returns.

3. **Moving Average Crossover Strategy (20%)**: Uses multiple moving average signals (5/20 day and 10/30 day crossovers) to identify trend direction.

4. **Risk Parity Strategy (15%)**: Allocates capital to equalize risk contribution from each asset, rather than capital contribution.

5. **Volume-Weighted Strategy (10%)**: Incorporates liquidity measures to favor more liquid assets.

6. **Trend Following Strategy (20%)**: Uses linear regression slope analysis to identify and allocate towards assets with stronger positive trends.

## Advanced Risk Management

The strategy implements several sophisticated risk management techniques:

1. **Adaptive Volatility Targeting**: Dynamically adjusts position sizes to target a specific portfolio volatility, which varies based on detected market regimes.

2. **Market Regime Detection**: Analyzes recent returns and volatility to determine market conditions on a scale from bearish (0) to bullish (1).

3. **Position Size Constraints**: Applies maximum (25%) and minimum (1%) position size limits to prevent concentration risk and eliminate negligible positions.

4. **Portfolio Smoothing**: Reduces turnover by exponentially smoothing allocation changes between rebalancing periods.

5. **Robust Error Handling**: Implements comprehensive error detection and fallback mechanisms to handle edge cases and ensure stability.

## Performance Optimization

The strategy specifically targets improvement across the competition's evaluation metrics:

- **Sharpe Ratio**: Through effective diversification and risk parity allocation
- **Sortino Ratio**: By adaptive volatility targeting and market regime detection
- **Max Drawdown**: Through dynamic risk allocation during adverse market conditions
- **Volatility**: Via minimum volatility component and volatility scaling
- **Annual Return**: Momentum and trend following components capture upside
- **Calmar Ratio**: Combination of drawdown reduction and return enhancement
- **Win Rate**: Moving average and trend following systems improve directionality

## Implementation

The strategy is implemented in a single Python class adhering to the competition requirements:
- Uses only allowed libraries (numpy, pandas, scipy)
- Handles all edge cases gracefully
- Maintains computational efficiency for real-time rebalancing

## Authors

Developed for the Amherst Quant Competition 2025.
