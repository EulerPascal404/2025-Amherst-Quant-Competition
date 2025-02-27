import numpy as np
import pandas as pd
import scipy
from collections import deque

class PortfolioStrategy:
    def __init__(self):
        """
        Initialize strategy parameters
        """
        # Strategy parameters - optimized for competition metrics
        self.lookback_window = 20  # Optimized lookback window
        self.volatility_window = 15
        self.ma_short_window = 5
        self.ma_long_window = 20
        self.volume_window = 10
        self.trend_window = 10
        
        # Strategy weights - tuned for the competition metrics
        self.momentum_weight = 0.25
        self.min_vol_weight = 0.20
        self.ma_weight = 0.15
        self.risk_parity_weight = 0.15
        self.volume_weight = 0.05
        self.trend_weight = 0.20
        
        # Position constraints
        self.max_position_size = 0.20  # Conservative position limit
        self.min_position_size = 0.005  # Allow smaller positions
        
        # Adaptive risk management parameters
        self.market_regime_window = 15
        self.vol_lookback = 15
        self.volatility_scale_factor = 0.12  # Target annual volatility
        self.market_regime_threshold = 0.015
        
        # Store historical data
        self.close_history = deque(maxlen=max(self.lookback_window, self.ma_long_window))
        self.high_history = deque(maxlen=self.volatility_window)
        self.low_history = deque(maxlen=self.volatility_window)
        self.volume_history = deque(maxlen=self.volume_window)
        
        # Day counter
        self.day_count = 0
        
        # Covariance matrix
        self.cov_matrix = None
        self.cov_lookback = 20
        self.returns_history = deque(maxlen=self.cov_lookback)
        
        # Previous weights
        self.prev_weights = None
        
        # Market regime tracking
        self.market_returns = deque(maxlen=self.market_regime_window)
        self.current_volatility = 0.01

    def calculate_momentum_weights(self, latest_close):
        """Calculate momentum-based weights"""
        if self.day_count < 2:
            return np.ones(self.n_assets) / self.n_assets
        
        # Convert history to array
        close_array = np.array(list(self.close_history))
        
        # Calculate momentum over multiple lookback periods
        momentum_5d = (latest_close - close_array[-5]) / close_array[-5] if len(close_array) >= 5 else np.zeros(self.n_assets)
        momentum_10d = (latest_close - close_array[-10]) / close_array[-10] if len(close_array) >= 10 else np.zeros(self.n_assets)
        momentum_20d = (latest_close - close_array[-self.lookback_window]) / close_array[-self.lookback_window] if len(close_array) >= self.lookback_window else np.zeros(self.n_assets)
        
        # Combine momentum signals with more weight to more recent
        momentum = 0.6 * momentum_5d + 0.3 * momentum_10d + 0.1 * momentum_20d
        
        # Only consider positive momentum
        weights = np.maximum(momentum, 0)
        
        # Normalize
        return self.normalize_weights(weights)

    def calculate_volatility_weights(self):
        """Calculate inverse volatility weights"""
        if self.day_count < self.volatility_window:
            return np.ones(self.n_assets) / self.n_assets
        
        # Calculate volatility using returns
        close_array = np.array(list(self.close_history)[-self.volatility_window:])
        if close_array.shape[0] > 1:
            returns = np.diff(close_array, axis=0) / close_array[:-1]
            volatility = np.std(returns, axis=0)
        else:
            return np.ones(self.n_assets) / self.n_assets
        
        # Handle near-zero volatility
        volatility = np.maximum(volatility, 1e-6)
        
        # Inverse volatility weighting
        inv_volatility = 1.0 / volatility
        
        # Normalize
        return self.normalize_weights(inv_volatility)
    
    def calculate_ma_crossover_weights(self):
        """Calculate moving average crossover weights"""
        if self.day_count < self.ma_long_window:
            return np.ones(self.n_assets) / self.n_assets
        
        # Convert history to array
        close_array = np.array(list(self.close_history))
        
        # Calculate moving averages
        ma_short = np.mean(close_array[-self.ma_short_window:], axis=0)
        ma_long = np.mean(close_array[-self.ma_long_window:], axis=0)
        
        # Crossover signal
        signal = ma_short > ma_long
        
        # Convert to weights
        weights = signal.astype(float)
        
        # Normalize
        return self.normalize_weights(weights)
    
    def calculate_risk_parity_weights(self):
        """Calculate risk parity weights"""
        if self.cov_matrix is None or self.day_count < self.cov_lookback:
            return np.ones(self.n_assets) / self.n_assets
        
        try:
            # Use volatility from covariance matrix diagonal
            volatilities = np.sqrt(np.diag(self.cov_matrix))
            volatilities = np.maximum(volatilities, 1e-6)
            
            # Inverse volatility for risk parity
            inv_vol = 1.0 / volatilities
            
            # Normalize
            return self.normalize_weights(inv_vol)
        except:
            return np.ones(self.n_assets) / self.n_assets
    
    def calculate_volume_weights(self):
        """Calculate volume-weighted allocation"""
        if self.day_count < self.volume_window:
            return np.ones(self.n_assets) / self.n_assets
        
        # Calculate average volume
        volume_array = np.array(list(self.volume_history))
        avg_volume = np.mean(volume_array, axis=0)
        
        # Normalize
        return self.normalize_weights(avg_volume)
    
    def calculate_trend_weights(self):
        """Calculate trend-based weights"""
        if self.day_count < self.trend_window:
            return np.ones(self.n_assets) / self.n_assets
        
        # Get price history
        close_array = np.array(list(self.close_history))
        
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
        prices = np.maximum(prices, 1e-6)  # Avoid division by zero
        normalized_slopes = slopes / prices
        
        # Only consider positive trends
        weights = np.maximum(normalized_slopes, 0)
        
        # Normalize
        return self.normalize_weights(weights)
    
    def update_covariance_matrix(self):
        """Update covariance matrix of returns"""
        if self.day_count < self.cov_lookback + 1:
            return
            
        try:
            # Get returns history
            returns_array = np.array(list(self.returns_history))
            
            # Calculate covariance with shrinkage
            sample_cov = np.cov(returns_array.T)
            target = np.diag(np.diag(sample_cov))
            shrinkage_factor = 0.3
            
            self.cov_matrix = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target
            self.cov_matrix += np.eye(self.n_assets) * 1e-6  # Numerical stability
        except:
            # Fallback to identity matrix
            self.cov_matrix = np.eye(self.n_assets)

    def detect_market_regime(self, daily_returns):
        """Detect market regime (bull/bear)"""
        if self.day_count < self.market_regime_window:
            return 0.5  # Neutral
        
        # Update market returns
        market_return = np.mean(daily_returns)
        self.market_returns.append(market_return)
        
        # Calculate market statistics
        market_returns_array = np.array(list(self.market_returns))
        mean_return = np.mean(market_returns_array)
        vol = np.std(market_returns_array) * np.sqrt(252)
        
        # Update volatility estimate
        self.current_volatility = vol
        
        # Calculate regime score
        return_score = (mean_return + self.market_regime_threshold) / (2 * self.market_regime_threshold)
        return_score = np.clip(return_score, 0, 1)
        
        return return_score
    
    def apply_volatility_targeting(self, weights, market_regime):
        """Adjust weights to target volatility"""
        if self.cov_matrix is None or self.day_count < self.vol_lookback:
            return weights
        
        try:
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
            
            # Ensure no weight exceeds 1.0
            if np.max(scaled_weights) > 1.0:
                scaled_weights = scaled_weights / np.max(scaled_weights)
                
            # Calculate cash allocation
            cash_weight = max(0, 1.0 - np.sum(scaled_weights))
            
            # Return normalized weights
            if cash_weight > 0:
                return scaled_weights
            else:
                return scaled_weights / np.sum(scaled_weights)
                
        except:
            return weights

    def apply_position_constraints(self, weights):
        """Apply position constraints"""
        # Maximum position size
        constrained_weights = np.minimum(weights, self.max_position_size)
        
        # Normalize
        constrained_weights = self.normalize_weights(constrained_weights)
        
        # Remove tiny positions
        constrained_weights[constrained_weights < self.min_position_size] = 0
        
        # Re-normalize
        return self.normalize_weights(constrained_weights)

    def smooth_weights(self, current_weights):
        """Smooth portfolio transitions"""
        if self.prev_weights is None:
            return current_weights
        
        # Exponential smoothing
        smoothing_factor = 0.7
        smoothed_weights = smoothing_factor * current_weights + (1 - smoothing_factor) * self.prev_weights
        
        # Normalize
        return self.normalize_weights(smoothed_weights)
    
    def normalize_weights(self, weights):
        """Normalize weights to sum to 1, handling edge cases"""
        if np.sum(weights) <= 1e-10:
            return np.ones(self.n_assets) / self.n_assets
        return weights / np.sum(weights)

    def allocate(self, market_data: dict) -> np.ndarray:
        """
        Main allocation function called by the backtester
        """
        try:
            # Access data
            closes = market_data['close']
            highs = market_data['high']
            lows = market_data['low']
            volumes = market_data['volume']
            
            # Handle bad data
            closes = np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0)
            highs = np.nan_to_num(highs, nan=0.0, posinf=0.0, neginf=0.0)
            lows = np.nan_to_num(lows, nan=0.0, posinf=0.0, neginf=0.0)
            volumes = np.nan_to_num(volumes, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Avoid division by zero
            closes = np.maximum(closes, 1e-6)
            highs = np.maximum(highs, 1e-6)
            lows = np.maximum(lows, 1e-6)
            
            # Store number of assets
            self.n_assets = len(closes)
            
            # Increment counter
            self.day_count += 1
            
            # Update history
            self.close_history.append(closes)
            self.high_history.append(highs)
            self.low_history.append(lows)
            self.volume_history.append(volumes)
            
            # Equal weighting for initial periods
            if self.day_count < 2:
                equal_weights = np.ones(self.n_assets) / self.n_assets
                self.prev_weights = equal_weights
                return equal_weights
                
            # Calculate daily returns and update history
            prev_close = np.array(list(self.close_history))[-2]
            daily_returns = (closes - prev_close) / prev_close
            self.returns_history.append(daily_returns)
            
            # Update covariance matrix
            self.update_covariance_matrix()
            
            # Detect market regime
            market_regime = self.detect_market_regime(daily_returns)
            
            # Calculate component strategy weights
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
                weights = np.ones(self.n_assets) / self.n_assets
            
            # Ensure non-negative
            weights = np.maximum(weights, 0)
            
            # Normalize
            weights = self.normalize_weights(weights)
            
            # Apply constraints
            weights = self.apply_position_constraints(weights)
            
            # Apply volatility targeting
            weights = self.apply_volatility_targeting(weights, market_regime)
            
            # Smooth transitions
            weights = self.smooth_weights(weights)
            
            # Store for next iteration
            self.prev_weights = weights.copy()
            
            return weights
            
        except Exception:
            # Fallback to equal weights
            equal_weights = np.ones(self.n_assets) / self.n_assets
            self.prev_weights = equal_weights
            return equal_weights 