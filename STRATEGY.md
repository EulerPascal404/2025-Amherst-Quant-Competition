# Algorithmic Trading Strategy Details

This document provides a comprehensive explanation of the strategies implemented in our algorithmic trading model for the Amherst Quant Competition 2025.

## Multi-Factor Approach

Our algorithm implements a weighted combination of six distinct strategies, each targeting different market inefficiencies:

### 1. Momentum Strategy (25%)

**Concept:** Momentum investing is based on the empirical observation that assets that have performed well in the recent past tend to continue performing well in the near future.

**Implementation Details:**
- We calculate momentum over multiple time horizons (5, 10, and 20 days) to capture both short and medium-term trends
- More weight is given to more recent performance (60% to 5-day momentum, 30% to 10-day, 10% to 20-day)
- Only positive momentum is considered (negative momentum stocks receive zero weight)
- The formula used is: `(current_price - price_n_days_ago) / price_n_days_ago`
- Weights are normalized to sum to 1

**Code Implementation:**
```python
momentum_5d = (latest_close - close_array[-5]) / close_array[-5]
momentum_10d = (latest_close - close_array[-10]) / close_array[-10]
momentum_20d = (latest_close - close_array[-self.lookback_window]) / close_array[-self.lookback_window]
        
# Combine momentum signals with more weight to more recent
momentum = 0.6 * momentum_5d + 0.3 * momentum_10d + 0.1 * momentum_20d
        
# Only consider positive momentum
weights = np.maximum(momentum, 0)
```

### 2. Minimum Volatility Strategy (20%)

**Concept:** Lower volatility stocks tend to deliver better risk-adjusted returns over time, a phenomenon known as the "low volatility anomaly."

**Implementation Details:**
- Volatility is calculated as the standard deviation of returns over a 15-day window
- Inverse volatility weighting gives higher allocations to less volatile assets
- This component helps improve the Sharpe and Sortino ratios
- Edge cases (zero or near-zero volatility) are handled to maintain numerical stability

**Code Implementation:**
```python
returns = np.diff(close_array, axis=0) / close_array[:-1]
volatility = np.std(returns, axis=0)
        
# Handle near-zero volatility
volatility = np.maximum(volatility, 1e-6)
        
# Inverse volatility weighting
inv_volatility = 1.0 / volatility
```

### 3. Moving Average Crossover Strategy (15%)

**Concept:** Moving average crossovers help identify trend changes and generate buy/sell signals when short-term averages cross long-term averages.

**Implementation Details:**
- We use a 5-day moving average as the short-term signal and a 20-day moving average as the long-term signal
- Assets where the short-term MA is above the long-term MA receive a positive weight
- This is a binary signal (either an asset receives weight or it doesn't)
- This component improves the win rate and helps identify trend changes

**Code Implementation:**
```python
ma_short = np.mean(close_array[-self.ma_short_window:], axis=0)
ma_long = np.mean(close_array[-self.ma_long_window:], axis=0)
        
# Crossover signal
signal = ma_short > ma_long
        
# Convert to weights
weights = signal.astype(float)
```

### 4. Risk Parity Strategy (15%)

**Concept:** Risk parity allocates capital to equalize risk contribution from each asset, rather than equal capital allocation.

**Implementation Details:**
- Uses the covariance matrix of returns to estimate volatility
- Allocates inversely proportional to volatility, so higher volatility assets receive less capital
- Calculation includes covariance shrinkage (30%) towards the diagonal to improve stability
- This approach helps reduce drawdowns and improve risk-adjusted returns

**Code Implementation:**
```python
# Calculate covariance with shrinkage
sample_cov = np.cov(returns_array.T)
target = np.diag(np.diag(sample_cov))
shrinkage_factor = 0.3
            
self.cov_matrix = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target

# Use volatility from covariance matrix diagonal
volatilities = np.sqrt(np.diag(self.cov_matrix))
volatilities = np.maximum(volatilities, 1e-6)
            
# Inverse volatility for risk parity
inv_vol = 1.0 / volatilities
```

### 5. Volume-Weighted Strategy (5%)

**Concept:** Higher trading volume often indicates better liquidity and potential for price movement.

**Implementation Details:**
- Calculates the average trading volume over a 10-day window
- Weights are proportional to volume (higher volume → higher weight)
- This component gives a small preference to more liquid assets
- Acts as a secondary factor to complement the other strategies

**Code Implementation:**
```python
volume_array = np.array(list(self.volume_history))
avg_volume = np.mean(volume_array, axis=0)
        
# Normalize
weights = avg_volume / np.sum(avg_volume)
```

### 6. Trend Following Strategy (20%)

**Concept:** Trend following identifies and capitalizes on persistent market trends using statistical techniques.

**Implementation Details:**
- Uses linear regression to calculate the slope of price movement over a 10-day window
- Slopes are normalized by price level to make them comparable across assets
- Only positive trends (upward slopes) are considered
- This component helps capture stronger trending assets

**Code Implementation:**
```python
# Calculate regression slopes
x = np.arange(self.trend_window)
slopes = np.zeros(self.n_assets)
        
# Linear regression for each asset
for i in range(self.n_assets):
    y = close_array[-self.trend_window:, i]
    try:
        slope, _ = np.polyfit(x, y, 1)
        slopes[i] = slope
    except:
        slopes[i] = 0
        
# Normalize slopes by price level
prices = close_array[-1]
prices = np.maximum(prices, 1e-6)
normalized_slopes = slopes / prices
        
# Only consider positive trends
weights = np.maximum(normalized_slopes, 0)
```

## Advanced Risk Management Components

Our strategy incorporates sophisticated risk management techniques:

### 1. Market Regime Detection

**Concept:** Market conditions vary between bullish and bearish regimes, each requiring different allocation approaches.

**Implementation Details:**
- Uses a 15-day window of average market returns to detect the current regime
- Calculates a numerical score between 0 (very bearish) and 1 (very bullish)
- This score is used to adjust the volatility targeting
- Returns and volatility both factor into the regime determination

**Code Implementation:**
```python
market_return = np.mean(daily_returns)
self.market_returns.append(market_return)
        
market_returns_array = np.array(list(self.market_returns))
mean_return = np.mean(market_returns_array)
vol = np.std(market_returns_array) * np.sqrt(252)
        
# Calculate regime score
return_score = (mean_return + self.market_regime_threshold) / (2 * self.market_regime_threshold)
return_score = np.clip(return_score, 0, 1)
```

### 2. Adaptive Volatility Targeting

**Concept:** Adjusting portfolio leverage to maintain a consistent level of risk across different market conditions.

**Implementation Details:**
- Target volatility varies based on the detected market regime
- In bullish regimes, a higher target volatility is accepted
- In bearish regimes, target volatility is reduced to preserve capital
- Scaling factor is limited to avoid extreme changes
- Portfolio weights are scaled up or down to reach the target volatility

**Code Implementation:**
```python
# Calculate portfolio variance
portfolio_variance = weights.T @ self.cov_matrix @ weights
            
# Annualized volatility
portfolio_vol = np.sqrt(portfolio_variance * 252)
            
# Target volatility based on market regime
adaptive_target_vol = self.volatility_scale_factor * (0.5 + 0.5 * market_regime)
            
# Scaling factor
if portfolio_vol > 1e-6:
    scaling_factor = adaptive_target_vol / portfolio_vol
else:
    scaling_factor = 1.0
                
# Limit extreme changes
scaling_factor = np.clip(scaling_factor, 0.5, 2.0)
            
# Scale weights
scaled_weights = weights * scaling_factor
```

### 3. Position Size Constraints

**Concept:** Limiting individual position sizes to prevent overconcentration in any single asset.

**Implementation Details:**
- Maximum position size capped at 20% to prevent concentration risk
- Minimum position size threshold of 0.5% eliminates negligible positions
- Ensures diversification while maintaining meaningful exposures
- Weights are renormalized after constraints are applied

**Code Implementation:**
```python
# Maximum position size
constrained_weights = np.minimum(weights, self.max_position_size)
        
# Normalize
constrained_weights = self.normalize_weights(constrained_weights)
        
# Remove tiny positions
constrained_weights[constrained_weights < self.min_position_size] = 0
```

### 4. Portfolio Smoothing

**Concept:** Reducing turnover by gradually transitioning between target weights.

**Implementation Details:**
- Uses exponential smoothing between current and previous weights
- Smoothing factor of 0.7 gives 70% weight to new allocation and 30% to previous allocation
- Reduces transaction costs and potential market impact
- Provides more stable allocations over time

**Code Implementation:**
```python
# Exponential smoothing
smoothing_factor = 0.7
smoothed_weights = smoothing_factor * current_weights + (1 - smoothing_factor) * self.prev_weights
```

## Performance Optimization

Our strategy is specifically optimized for the competition's evaluation metrics:

1. **Sharpe Ratio**: Enhanced through the combination of minimum volatility and risk parity components, which focus on risk-adjusted returns.

2. **Sortino Ratio**: Improved through adaptive volatility targeting and downside risk management in the market regime detection system.

3. **Max Drawdown**: Reduced by dynamically scaling back risk during adverse market conditions and increasing exposure to lower volatility assets.

4. **Volatility**: Controlled through inverse volatility weighting and adaptive volatility targeting.

5. **Annual Return**: Maximized via momentum and trend following components that capture upward price movements.

6. **Calmar Ratio**: Optimized by balancing return generation (momentum, trend) with drawdown reduction (volatility targeting).

7. **Win Rate**: Improved through moving average crossovers and trend identification that increase the proportion of winning trades.

## Robust Implementation

The strategy implementation includes several features to ensure stability and handle edge cases:

1. **Comprehensive error handling** with try/except blocks and fallback mechanisms
2. **Data quality checks** including handling of NaN, infinity, and zero values
3. **Normalization functions** to ensure weights always sum to 1
4. **Adaptive parameters** that adjust based on market conditions
5. **Memory-efficient data storage** using deques with fixed maximum length

These features ensure the strategy performs reliably across diverse market conditions and various asset types. 