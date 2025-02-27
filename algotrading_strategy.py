import numpy as np
import pandas as pd
import scipy
from collections import deque

class PortfolioStrategy:
    def __init__(self):
        """
        Initialize strategy parameters
        """
        # Strategy parameters
        self.lookback_window = 30  # Days to look back for momentum calculation
        self.volatility_window = 20  # Days to calculate volatility
        self.ma_short_window = 10  # Short-term moving average window
        self.ma_long_window = 30  # Long-term moving average window
        self.volume_window = 10  # Window for volume analysis
        self.trend_window = 15  # Window for trend analysis
        
        # Strategy weights
        self.momentum_weight = 0.20
        self.min_vol_weight = 0.15
        self.ma_weight = 0.20
        self.risk_parity_weight = 0.15
        self.volume_weight = 0.10
        self.trend_weight = 0.20
        
        # Position constraints
        self.max_position_size = 0.25  # Maximum weight for any single asset
        self.min_position_size = 0.01  # Minimum weight to avoid very small positions
        
        # Adaptive risk management parameters
        self.market_regime_window = 20  # Window for detecting market regime
        self.vol_lookback = 20  # Lookback for calculating market volatility
        self.volatility_scale_factor = 0.15  # Target annualized volatility
        self.market_regime_threshold = 0.02  # Threshold for regime change detection
        
        # Store historical data
        self.close_history = deque(maxlen=max(self.lookback_window, self.ma_long_window))
        self.high_history = deque(maxlen=self.volatility_window)
        self.low_history = deque(maxlen=self.volatility_window)
        self.volume_history = deque(maxlen=self.volume_window)
        
        # Tracking day count
        self.day_count = 0
        
        # To store calculated covariance matrix
        self.cov_matrix = None
        self.cov_lookback = 30
        self.returns_history = deque(maxlen=self.cov_lookback)
        
        # Store previous weights for smoothing
        self.prev_weights = None
        
        # Performance tracking
        self.portfolio_values = deque(maxlen=60)  # Track last 60 days of portfolio value
        self.market_returns = deque(maxlen=self.market_regime_window)  # Store market returns for regime detection
        self.current_volatility = 0.01  # Initialize to a low value

    def calculate_momentum_weights(self, latest_close):
        """
        Calculate momentum-based weights using the full lookback window
        """
        # If we don't have enough history, return equal weights
        if self.day_count < self.lookback_window:
            return np.ones(self.n_assets) / self.n_assets
        
        # Convert history to array
        close_array = np.array(list(self.close_history))
        
        # Calculate momentum over multiple lookback periods for robustness
        momentum_10d = (latest_close - close_array[-10]) / close_array[-10] if len(close_array) >= 10 else np.zeros(self.n_assets)
        momentum_20d = (latest_close - close_array[-20]) / close_array[-20] if len(close_array) >= 20 else np.zeros(self.n_assets)
        momentum_30d = (latest_close - close_array[-self.lookback_window]) / close_array[-self.lookback_window]
        
        # Combine momentum signals with more weight to recent momentum
        momentum = 0.5 * momentum_10d + 0.3 * momentum_20d + 0.2 * momentum_30d
        
        # Only consider positive momentum
        weights = np.maximum(momentum, 0)
        
        # If all weights are zero, return equal weights
        if np.sum(weights) <= 1e-10:
            return np.ones(self.n_assets) / self.n_assets
            
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        return weights

    def calculate_volatility_weights(self):
        """
        Calculate inverse volatility weights
        """
        # If we don't have enough history, return equal weights
        if self.day_count < self.volatility_window:
            return np.ones(self.n_assets) / self.n_assets
        
        # Convert history to arrays
        close_array = np.array(list(self.close_history)[-self.volatility_window:])
        
        # Calculate price-based volatility
        if close_array.shape[0] > 1:
            returns = np.diff(close_array, axis=0) / close_array[:-1]
            volatility = np.std(returns, axis=0)
        else:
            return np.ones(self.n_assets) / self.n_assets
        
        # Handle zero or very small volatility
        volatility = np.maximum(volatility, 1e-6)
        
        # Use inverse volatility for weighting (lower volatility -> higher weight)
        inv_volatility = 1.0 / volatility
        
        # Normalize weights to sum to 1
        weights = inv_volatility / np.sum(inv_volatility)
        return weights
    
    def calculate_ma_crossover_weights(self):
        """
        Calculate weights based on moving average crossovers
        """
        # If we don't have enough history, return equal weights
        if self.day_count < self.ma_long_window:
            return np.ones(self.n_assets) / self.n_assets
        
        # Convert history to array
        close_array = np.array(list(self.close_history))
        
        # Calculate multiple moving averages for robustness
        ma_5 = np.mean(close_array[-5:], axis=0) if len(close_array) >= 5 else close_array[-1]
        ma_10 = np.mean(close_array[-10:], axis=0) if len(close_array) >= 10 else close_array[-1]
        ma_20 = np.mean(close_array[-20:], axis=0) if len(close_array) >= 20 else close_array[-1]
        ma_30 = np.mean(close_array[-30:], axis=0) if len(close_array) >= 30 else close_array[-1]
        
        # Calculate crossover signals
        signal1 = ma_5 > ma_20  # Short-term crossover
        signal2 = ma_10 > ma_30  # Long-term crossover
        
        # Combine signals
        signal = signal1.astype(float) + signal2.astype(float)
        
        # If no signals, return equal weights
        if not np.any(signal):
            return np.ones(self.n_assets) / self.n_assets
        
        # Normalize
        weights = signal / np.sum(signal)
        
        return weights
    
    def calculate_risk_parity_weights(self):
        """
        Calculate risk parity weights (simplified implementation)
        """
        # If we don't have enough history or covariance matrix, return equal weights
        if self.cov_matrix is None or self.day_count < self.cov_lookback:
            return np.ones(self.n_assets) / self.n_assets
        
        try:
            # Calculate risk contributions
            volatilities = np.sqrt(np.diag(self.cov_matrix))
            
            # Handle near-zero volatilities
            volatilities = np.maximum(volatilities, 1e-6)
            
            # Inverse of volatility for risk parity (equal risk contribution)
            inv_vol = 1.0 / volatilities
            
            # Normalize weights to sum to 1
            weights = inv_vol / np.sum(inv_vol)
            
        except Exception:
            # Fallback to equal weights on numerical issues
            weights = np.ones(self.n_assets) / self.n_assets
        
        return weights
    
    def calculate_volume_weights(self):
        """
        Calculate weights based on trading volume (higher volume -> higher weight)
        """
        # If we don't have enough history, return equal weights
        if self.day_count < self.volume_window:
            return np.ones(self.n_assets) / self.n_assets
        
        # Convert history to array
        volume_array = np.array(list(self.volume_history))
        
        # Calculate average volume
        avg_volume = np.mean(volume_array, axis=0)
        
        # Handle zero volume
        if np.sum(avg_volume) <= 1e-10:
            return np.ones(self.n_assets) / self.n_assets
        
        # Normalize weights to sum to 1
        weights = avg_volume / np.sum(avg_volume)
        
        return weights
    
    def calculate_trend_weights(self):
        """
        Calculate weights based on trend analysis
        """
        # If we don't have enough history, return equal weights
        if self.day_count < self.trend_window:
            return np.ones(self.n_assets) / self.n_assets
        
        # Convert history to array
        close_array = np.array(list(self.close_history))
        
        # Calculate linear regression slope for each asset
        x = np.arange(self.trend_window)
        slopes = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            y = close_array[-self.trend_window:, i]
            try:
                slope, _ = np.polyfit(x, y, 1)
                slopes[i] = slope
            except:
                slopes[i] = 0
        
        # Normalize slopes relative to price level
        normalized_slopes = slopes / np.mean(close_array[-1])
        
        # Only consider positive trends
        weights = np.maximum(normalized_slopes, 0)
        
        # If all weights are zero, return equal weights
        if np.sum(weights) <= 1e-10:
            return np.ones(self.n_assets) / self.n_assets
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def update_covariance_matrix(self):
        """
        Update the covariance matrix of returns
        """
        if self.day_count < self.cov_lookback + 1:
            return
            
        try:
            # Convert returns history to array
            returns_array = np.array(list(self.returns_history))
            
            # Calculate covariance matrix with regularization (shrinkage)
            sample_cov = np.cov(returns_array.T)
            
            # Simple shrinkage towards diagonal
            target = np.diag(np.diag(sample_cov))
            shrinkage_factor = 0.3  # Adjust based on empirical performance
            
            self.cov_matrix = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target
            
            # Add small value to diagonal for numerical stability
            self.cov_matrix += np.eye(self.n_assets) * 1e-6
        except:
            # If covariance calculation fails, use identity matrix
            self.cov_matrix = np.eye(self.n_assets)

    def apply_position_constraints(self, weights):
        """
        Apply position size constraints
        """
        # Apply maximum position constraint
        constrained_weights = np.minimum(weights, self.max_position_size)
        
        # Normalize to sum to 1
        if np.sum(constrained_weights) > 0:
            constrained_weights = constrained_weights / np.sum(constrained_weights)
        else:
            constrained_weights = np.ones(self.n_assets) / self.n_assets
        
        # Eliminate very small positions
        constrained_weights[constrained_weights < self.min_position_size] = 0
        
        # Renormalize
        if np.sum(constrained_weights) > 0:
            constrained_weights = constrained_weights / np.sum(constrained_weights)
        else:
            constrained_weights = np.ones(self.n_assets) / self.n_assets
        
        return constrained_weights

    def smooth_weights(self, current_weights):
        """
        Smooth portfolio transitions to reduce turnover
        """
        if self.prev_weights is None:
            return current_weights
        
        # Apply exponential smoothing
        smoothing_factor = 0.7  # Higher value gives more weight to new allocation
        smoothed_weights = smoothing_factor * current_weights + (1 - smoothing_factor) * self.prev_weights
        
        # Normalize to sum to 1
        smoothed_weights = smoothed_weights / np.sum(smoothed_weights)
        
        return smoothed_weights
    
    def detect_market_regime(self, daily_returns):
        """
        Detect market regime (bull or bear) based on recent returns and volatility
        Returns a value between 0 (bearish) and 1 (bullish)
        """
        # If we don't have enough history, assume neutral market
        if self.day_count < self.market_regime_window:
            return 0.5
        
        # Update market returns history
        # Use average return across all assets as a proxy for market return
        market_return = np.mean(daily_returns)
        self.market_returns.append(market_return)
        
        # Calculate market statistics
        market_returns_array = np.array(list(self.market_returns))
        mean_return = np.mean(market_returns_array)
        vol = np.std(market_returns_array) * np.sqrt(252)  # Annualized volatility
        
        # Update current volatility estimate
        self.current_volatility = vol
        
        # Calculate market regime score based on return and volatility
        # High returns and low volatility = bullish (1.0)
        # Low returns and high volatility = bearish (0.0)
        return_score = (mean_return + self.market_regime_threshold) / (2 * self.market_regime_threshold)
        return_score = np.clip(return_score, 0, 1)
        
        # Return the regime score (0 = very bearish, 1 = very bullish)
        return return_score
    
    def apply_volatility_targeting(self, weights, market_regime):
        """
        Adjust position sizes to target a specific portfolio volatility
        based on current market conditions
        """
        if self.cov_matrix is None or self.day_count < self.vol_lookback:
            return weights
        
        try:
            # Calculate portfolio variance
            portfolio_variance = weights.T @ self.cov_matrix @ weights
            
            # Annualize volatility (assuming daily data, multiply by sqrt(252))
            portfolio_vol = np.sqrt(portfolio_variance * 252)
            
            # Adjust target volatility based on market regime
            # More bullish = higher target volatility
            adaptive_target_vol = self.volatility_scale_factor * (0.5 + 0.5 * market_regime)
            
            # Calculate scaling factor to reach target vol
            if portfolio_vol > 1e-6:
                scaling_factor = adaptive_target_vol / portfolio_vol
            else:
                scaling_factor = 1.0
                
            # Limit scaling factor to avoid extreme changes
            scaling_factor = np.clip(scaling_factor, 0.5, 2.0)
            
            # Scale all weights by the same factor
            scaled_weights = weights * scaling_factor
            
            # Ensure no weight exceeds 1.0
            if np.max(scaled_weights) > 1.0:
                scaled_weights = scaled_weights / np.max(scaled_weights)
                
            # Calculate cash weight (1 - sum of allocations)
            cash_weight = max(0, 1.0 - np.sum(scaled_weights))
            
            # If we have cash, we need to renormalize the asset weights
            if cash_weight > 0:
                return scaled_weights
            else:
                # Normalize to sum to 1
                return scaled_weights / np.sum(scaled_weights)
                
        except:
            # Return original weights if calculation fails
            return weights

    def allocate(self, market_data: dict) -> np.ndarray:
        """
        market_data: Dictionary containing numpy arrays for:
            - 'open': Opening prices
            - 'close': Closing prices
            - 'high': High prices
            - 'low': Low prices
            - 'volume': Trading volumes
        for the current trading day
        
        Returns: numpy array of portfolio weights
        """
        try:
            # Access different data types
            closes = market_data['close']
            highs = market_data['high']
            lows = market_data['low']
            volumes = market_data['volume']
            
            # Handle NaN or infinity values
            closes = np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0)
            highs = np.nan_to_num(highs, nan=0.0, posinf=0.0, neginf=0.0)
            lows = np.nan_to_num(lows, nan=0.0, posinf=0.0, neginf=0.0)
            volumes = np.nan_to_num(volumes, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Replace zeros with small positive values to avoid division by zero
            closes = np.maximum(closes, 1e-6)
            highs = np.maximum(highs, 1e-6)
            lows = np.maximum(lows, 1e-6)
            
            # Store number of assets
            self.n_assets = len(closes)
            
            # Increment day counter
            self.day_count += 1
            
            # Update history
            self.close_history.append(closes)
            self.high_history.append(highs)
            self.low_history.append(lows)
            self.volume_history.append(volumes)
            
            # Equal weighting for initial periods until we have enough history
            if self.day_count < 2:
                equal_weights = np.ones(self.n_assets) / self.n_assets
                self.prev_weights = equal_weights
                return equal_weights
                
            # Calculate daily returns and update returns history
            prev_close = np.array(list(self.close_history))[-2]
            daily_returns = (closes - prev_close) / prev_close
            self.returns_history.append(daily_returns)
            
            # Update covariance matrix
            self.update_covariance_matrix()
            
            # Detect market regime
            market_regime = self.detect_market_regime(daily_returns)
            
            # Calculate weights for each component strategy
            try:
                momentum_weights = self.calculate_momentum_weights(closes)
                volatility_weights = self.calculate_volatility_weights()
                ma_weights = self.calculate_ma_crossover_weights()
                risk_parity_weights = self.calculate_risk_parity_weights()
                volume_weights = self.calculate_volume_weights()
                trend_weights = self.calculate_trend_weights()
                
                # Combine strategies
                weights = (
                    self.momentum_weight * momentum_weights + 
                    self.min_vol_weight * volatility_weights +
                    self.ma_weight * ma_weights +
                    self.risk_parity_weight * risk_parity_weights +
                    self.volume_weight * volume_weights +
                    self.trend_weight * trend_weights
                )
            except Exception:
                # Fall back to equal weighting if any strategy fails
                weights = np.ones(self.n_assets) / self.n_assets
            
            # Ensure weights are non-negative
            weights = np.maximum(weights, 0)
            
            # Normalize to sum to 1
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(self.n_assets) / self.n_assets
            
            # Apply position constraints
            weights = self.apply_position_constraints(weights)
            
            # Apply volatility targeting based on market regime
            weights = self.apply_volatility_targeting(weights, market_regime)
            
            # Smooth weights to reduce turnover
            weights = self.smooth_weights(weights)
            
            # Store weights for next iteration
            self.prev_weights = weights.copy()
            
            return weights
            
        except Exception:
            # Fallback to equal weights for any unexpected errors
            equal_weights = np.ones(self.n_assets) / self.n_assets
            self.prev_weights = equal_weights
            return equal_weights 