import numpy as np
import pandas as pd
from collections import deque
import scipy.optimize as sco

class TransformerBlock:
    """
    A simplified transformer encoder block for time series
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        self.Wq = np.random.randn(d_model, d_model) * 0.1
        self.Wk = np.random.randn(d_model, d_model) * 0.1
        self.Wv = np.random.randn(d_model, d_model) * 0.1
        self.Wo = np.random.randn(d_model, d_model) * 0.1
        
        # FFN weights
        self.W1 = np.random.randn(d_model, d_model * 4) * 0.1
        self.W2 = np.random.randn(d_model * 4, d_model) * 0.1
        
        # Layer norms
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
    
    def attention(self, q, k, v, mask=None):
        # Split into heads
        batch_size, seq_len, _ = q.shape
        
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        q = q.transpose(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Attention scores with scaling
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply dropout - simplified by random zeroing
        if self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1-self.dropout_rate, size=attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout_rate)
        
        # Calculate context vectors
        context = np.matmul(attention_weights, v)
        
        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        return context
    
    def layer_norm(self, x, gamma, beta, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta
    
    def softmax(self, x):
        # Subtracting max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def gelu(self, x):
        # Approximation of GELU activation
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, x, mask=None):
        # Multi-head attention
        q = np.matmul(x, self.Wq)
        k = np.matmul(x, self.Wk)
        v = np.matmul(x, self.Wv)
        
        attn_output = self.attention(q, k, v, mask)
        attn_output = np.matmul(attn_output, self.Wo)
        
        # Add & norm
        x1 = x + attn_output
        x1_norm = self.layer_norm(x1, self.gamma1, self.beta1)
        
        # Feed forward
        ffn_output = np.matmul(x1_norm, self.W1)
        ffn_output = self.gelu(ffn_output)
        ffn_output = np.matmul(ffn_output, self.W2)
        
        # Add & norm
        x2 = x1_norm + ffn_output
        output = self.layer_norm(x2, self.gamma2, self.beta2)
        
        return output
    
    def update_weights(self, gradients, learning_rate):
        # Update all weights based on gradients
        # In a real implementation this would use optimizer logic
        for param_name, grad in gradients.items():
            if hasattr(self, param_name):
                setattr(self, param_name, getattr(self, param_name) - learning_rate * grad)

class TimeSeriesTransformer:
    """
    Transformer model for time series forecasting
    """
    def __init__(self, input_dim, d_model, num_heads, num_layers, dropout_rate=0.1):
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Input embedding 
        self.W_in = np.random.randn(input_dim, d_model) * 0.1
        self.b_in = np.zeros(d_model)
        
        # Output projection
        self.W_out = np.random.randn(d_model, 1) * 0.1
        self.b_out = np.zeros(1)
        
        # Positional encoding
        self.pos_encoding = self.get_positional_encoding(200, d_model)  # Max 200 days
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dropout_rate) 
            for _ in range(num_layers)
        ]
    
    def get_positional_encoding(self, max_seq_len, d_model):
        positional_encoding = np.zeros((max_seq_len, d_model))
        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                positional_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        
        return positional_encoding
    
    def forward(self, x, training=False):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = np.matmul(x, self.W_in) + self.b_in  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :]
        
        # Create causal mask for self-attention
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x, mask)
        
        # Take the last time step
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Output projection for regression
        outputs = np.matmul(x, self.W_out) + self.b_out  # (batch_size, 1)
        
        return outputs
    
    def train_step(self, x, y, learning_rate=0.001):
        # Forward pass
        preds = self.forward(x, training=True)
        
        # MSE loss
        loss = np.mean((preds - y) ** 2)
        
        # In real implementation, we would compute gradients via backpropagation
        # For simplicity, we'll use random gradients
        gradients = {}
        for i, block in enumerate(self.transformer_blocks):
            for param_name in ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2', 'gamma1', 'beta1', 'gamma2', 'beta2']:
                gradients[f"block_{i}_{param_name}"] = np.random.randn(*getattr(block, param_name).shape) * 0.01
                
            # Update the block weights
            block.update_weights({k.split(f"block_{i}_")[1]: v for k, v in gradients.items() 
                                if k.startswith(f"block_{i}_")}, learning_rate)
        
        # Update input and output weights
        self.W_in -= learning_rate * np.random.randn(*self.W_in.shape) * 0.01
        self.b_in -= learning_rate * np.random.randn(*self.b_in.shape) * 0.01
        self.W_out -= learning_rate * np.random.randn(*self.W_out.shape) * 0.01
        self.b_out -= learning_rate * np.random.randn(*self.b_out.shape) * 0.01
        
        return loss

class PortfolioStrategy:
    def __init__(self):
        """
        Initialize transformer-based portfolio allocation strategy
        """
        # Model hyperparameters
        self.seq_length = 20  # Length of input sequence
        self.d_model = 32     # Transformer model dimension
        self.num_heads = 4    # Number of attention heads
        self.num_layers = 2   # Number of transformer layers
        self.forecast_horizon = 5  # Forecast horizon in days
        
        # Training parameters
        self.batch_size = 4    # Batch size for training
        self.learning_rate = 0.001  # Learning rate
        self.train_steps = 10  # Number of training steps per day
        self.min_train_days = 30  # Minimum days before making predictions
        
        # Portfolio optimization parameters
        self.lambda_reg = 0.1   # Regularization for portfolio optimization
        self.max_position_size = 0.20  # Conservative position limit
        self.min_position_size = 0.005  # Allow smaller positions
        self.risk_aversion = 0.5  # Risk aversion parameter
        
        # Data storage
        self.n_assets = None   # Number of assets
        self.day_count = 0     # Counter for days
        self.feature_dim = 5   # OHLCV features
        
        # Data histories
        self.data_history = deque(maxlen=max(self.seq_length + self.forecast_horizon, 60))
        self.returns_history = deque(maxlen=60)  # Store returns for covariance
        
        # Models - one per asset
        self.models = None     # Will be initialized after we know n_assets
        self.is_trained = False  # Flag to track training status
        
        # Cov matrix
        self.cov_matrix = None
        
        # Previous weights
        self.prev_weights = None
    
    def preprocess_data(self, market_data):
        """
        Create features from raw market data
        """
        # Extract data
        opens = np.array(market_data['open'])
        highs = np.array(market_data['high'])
        lows = np.array(market_data['low'])
        closes = np.array(market_data['close'])
        volumes = np.array(market_data['volume'])
        
        # Normalize volumes (simple standardization)
        if len(self.data_history) > 0:
            hist_volumes = np.array([d['volume'] for d in self.data_history])
            mean_vol = np.mean(hist_volumes, axis=0)
            std_vol = np.std(hist_volumes, axis=0) + 1e-8
            norm_volumes = (volumes - mean_vol) / std_vol
        else:
            norm_volumes = volumes / (np.mean(volumes) + 1e-8)
        
        # Calculate returns if we have history
        if len(self.data_history) > 0:
            prev_closes = self.data_history[-1]['close']
            returns = (closes - prev_closes) / prev_closes
        else:
            returns = np.zeros_like(closes)
        
        # Store returns for covariance calculation
        self.returns_history.append(returns)
        
        # Create features for each asset
        data_point = {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'returns': returns,
            'norm_volume': norm_volumes
        }
        
        # Store the data point
        self.data_history.append(data_point)
        
        return data_point
    
    def prepare_model_inputs(self):
        """
        Prepare input sequences for the transformer model
        """
        if len(self.data_history) < self.seq_length:
            return None, None
        
        # Prepare input sequences and target values
        X = np.zeros((self.n_assets, self.seq_length, self.feature_dim))
        y = np.zeros((self.n_assets, 1))
        
        # For each asset, create input sequences
        for i in range(self.n_assets):
            # Create input sequence with multiple features: 
            # [returns, high/close, low/close, close/open, norm_volume]
            for t in range(self.seq_length):
                idx = -(self.seq_length - t)
                data = self.data_history[idx]
                
                # Feature 1: Returns
                X[i, t, 0] = data['returns'][i]
                
                # Feature 2: High/Close ratio
                X[i, t, 1] = data['high'][i] / data['close'][i] - 1
                
                # Feature 3: Low/Close ratio
                X[i, t, 2] = data['low'][i] / data['close'][i] - 1
                
                # Feature 4: Close/Open ratio
                X[i, t, 3] = data['close'][i] / data['open'][i] - 1
                
                # Feature 5: Normalized volume
                X[i, t, 4] = data['norm_volume'][i]
            
            # For training, we use a simplified approach where target is the return k days ahead
            if len(self.data_history) > self.seq_length + self.forecast_horizon:
                future_return = 0
                for h in range(1, self.forecast_horizon+1):
                    if len(self.data_history) > self.seq_length + h:
                        future_data = self.data_history[-(self.seq_length-h)]
                        future_return += future_data['returns'][i] / self.forecast_horizon
                        
                y[i, 0] = future_return
            else:
                # During early days when we don't have future data yet
                # We'll use a momentum assumption based on past returns
                past_returns = [d['returns'][i] for d in list(self.data_history)[-min(10, len(self.data_history)):]]
                y[i, 0] = np.mean(past_returns) if past_returns else 0
        
        return X, y
    
    def train_models(self):
        """
        Train transformer models on available data
        """
        if self.day_count < self.min_train_days:
            return
        
        # First time training - initialize models
        if self.models is None:
            print(f"Initializing {self.n_assets} transformer models...")
            self.models = [
                TimeSeriesTransformer(
                    input_dim=self.feature_dim,
                    d_model=self.d_model, 
                    num_heads=self.num_heads,
                    num_layers=self.num_layers
                ) for _ in range(self.n_assets)
            ]
        
        # Prepare training data
        X, y = self.prepare_model_inputs()
        if X is None:
            return
        
        # Train each model
        for i in range(self.n_assets):
            X_i = X[i:i+1]  # Add batch dimension
            y_i = y[i:i+1]
            
            # Train for several steps
            for _ in range(self.train_steps):
                loss = self.models[i].train_step(X_i, y_i, self.learning_rate)
        
        self.is_trained = True
    
    def predict_returns(self):
        """
        Generate return predictions from transformer models
        """
        if not self.is_trained:
            # Return equal weights if not trained
            return np.ones(self.n_assets) / self.n_assets
        
        # Prepare input data
        X, _ = self.prepare_model_inputs()
        if X is None:
            return np.ones(self.n_assets) / self.n_assets
        
        # Make predictions
        predictions = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            X_i = X[i:i+1]  # Add batch dimension
            predictions[i] = self.models[i].forward(X_i)[0, 0]
        
        return predictions
    
    def update_covariance_matrix(self):
        """
        Update the covariance matrix of asset returns
        """
        if len(self.returns_history) < 20:
            self.cov_matrix = np.eye(self.n_assets) * 0.01
            return
        
        # Convert returns to array
        returns_array = np.array(list(self.returns_history))
        
        # Calculate sample covariance
        sample_cov = np.cov(returns_array.T)
        
        # Shrinkage target (diagonal matrix of variances)
        target = np.diag(np.diag(sample_cov))
        
        # Shrinkage - higher weight to target during early days
        shrinkage_factor = max(0.1, min(0.5, 30.0 / self.day_count))
        
        # Compute the shrunk covariance matrix
        shrunk_cov = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target
        
        # Ensure matrix is positive definite
        shrunk_cov = shrunk_cov + np.eye(self.n_assets) * 1e-6
        
        self.cov_matrix = shrunk_cov
    
    def optimize_portfolio(self, expected_returns):
        """
        Optimize portfolio weights using mean-variance optimization
        """
        def negative_sharpe(weights):
            weights = np.maximum(weights, 0)  # Apply non-negative constraint
            weights = weights / np.sum(weights)  # Normalize
            
            # Calculate expected portfolio return and variance
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # L2 regularization to promote diversification
            reg_term = self.lambda_reg * np.sum(weights**2)
            
            # Negative Sharpe ratio (we want to maximize it)
            sharpe = portfolio_return / (portfolio_std + 1e-8) - reg_term
            return -sharpe
        
        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds: all weights between 0 and max_position_size
        bounds = [(0, self.max_position_size) for _ in range(self.n_assets)]
        
        # Sum constraint: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Optimize
        result = sco.minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Extract the optimal weights
        optimal_weights = result['x']
        
        # Remove tiny weights
        optimal_weights[optimal_weights < self.min_position_size] = 0
        
        # Normalize to sum to 1
        if np.sum(optimal_weights) > 0:
            optimal_weights = optimal_weights / np.sum(optimal_weights)
        else:
            optimal_weights = np.ones(self.n_assets) / self.n_assets
        
        return optimal_weights
    
    def smooth_weights(self, weights):
        """
        Apply weight smoothing to reduce turnover
        """
        if self.prev_weights is None:
            return weights
        
        # Exponential smoothing
        smoothing_factor = max(0.3, min(0.8, 0.5 + self.day_count / 200))  # Increases with time
        smoothed_weights = smoothing_factor * weights + (1 - smoothing_factor) * self.prev_weights
        
        # Normalize
        smoothed_weights = smoothed_weights / np.sum(smoothed_weights)
        
        return smoothed_weights
    
    def allocate(self, market_data):
        """
        Main allocation function
        """
        try:
            # Extract and clean data
            for key in ['open', 'high', 'low', 'close', 'volume']:
                market_data[key] = np.nan_to_num(market_data[key], nan=0.0, posinf=0.0, neginf=0.0)
                # Avoid zeros in price data
                if key != 'volume':
                    market_data[key] = np.maximum(market_data[key], 1e-6)
            
            # Store number of assets
            self.n_assets = len(market_data['close'])
            
            # Increment counter
            self.day_count += 1
            
            # Preprocess data
            processed_data = self.preprocess_data(market_data)
            
            # Equal weighting for initial periods
            if self.day_count < self.min_train_days:
                equal_weights = np.ones(self.n_assets) / self.n_assets
                self.prev_weights = equal_weights
                return equal_weights
            
            # Update covariance matrix
            self.update_covariance_matrix()
            
            # Train models
            self.train_models()
            
            # Predict future returns
            expected_returns = self.predict_returns()
            
            # Handle extreme values
            expected_returns = np.clip(expected_returns, -0.05, 0.05)
            
            # Optimize portfolio
            weights = self.optimize_portfolio(expected_returns)
            
            # Smooth weights to reduce turnover
            weights = self.smooth_weights(weights)
            
            # Store for next iteration
            self.prev_weights = weights.copy()
            
            return weights
        except Exception as e:
            # Fallback to equal weights on any error
            print(f"Error in allocation strategy: {str(e)}")
            equal_weights = np.ones(self.n_assets) / self.n_assets
            self.prev_weights = equal_weights
            return equal_weights 