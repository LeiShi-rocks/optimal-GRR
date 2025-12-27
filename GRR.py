"""
General Regression Estimators (GRR) for causal inference in network settings.

This module implements general regression estimators that minimize a weighted
L2 loss for estimating treatment effects:

    min_{theta1, theta2} res' @ Kn @ res
    
where res = dHY - f1(X; theta1) * tilde1_d1 + f2(X; theta2) * tilde1_d2

IMPORTANT NOTE ON Kn:
    Kn = (G @ G.T > 0) is a binary indicator matrix for neighborhood overlap.
    This matrix is generally NOT positive semi-definite (PSD). Our approaches:
    
    - Linear models (OLS, Ridge): Direct solution (Z'KnZ)^{-1}Z'Kn y works
      regardless of PSD property as long as the system is solvable.
    
    - LASSO/ElasticNet: Use eigendecomposition with |eigenvalues| as an
      approximation to transform to standard form.
    
    - Tree-based models: Use diagonal approximation (row sums as sample weights).
    
    - Neural networks (PyTorch): Direct gradient descent on res'Kn res works
      fine - no transformation needed.

Supported function classes:
- Linear: OLS, Ridge, LASSO, Elastic Net
- Tree-based: Random Forest, Gradient Boosting, XGBoost (via sklearn interface)
- Neural Networks: MLP (via PyTorch)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from abc import ABC, abstractmethod

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.base import BaseEstimator, RegressorMixin, clone
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# Input Preparation
# =============================================================================

def prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, eps=1e-12):
    """
    Prepare inputs for GRR estimation.
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        X: covariate matrix (n x p)
        d1, d2: exposure levels to compare
        pi_id_all: propensity score matrix (n x k)
        eps: small constant for numerical stability
    
    Returns:
        dict with keys:
            hat1_d1, hat1_d2: IPW weights (n,)
            tilde1_d1, tilde1_d2: demeaned weights (n,)
            hatY_d1, hatY_d2: weighted outcomes (n,)
            dHY: difference in weighted outcomes (n,)
    """
    pi1 = pi_id_all[:, d1]
    pi2 = pi_id_all[:, d2]
    
    # IPW weights: hat1_d = I(D=d) / pi_d
    hat1_d1 = (D == d1).astype(float) / (pi1 + eps)
    hat1_d2 = (D == d2).astype(float) / (pi2 + eps)
    
    # Demeaned weights: tilde1_d = hat1_d - 1
    tilde1_d1 = hat1_d1 - 1.0
    tilde1_d2 = hat1_d2 - 1.0
    
    # Weighted outcomes: hatY_d = hat1_d * Y
    hatY_d1 = hat1_d1 * Y
    hatY_d2 = hat1_d2 * Y
    
    # Difference in weighted outcomes
    dHY = hatY_d1 - hatY_d2
    
    return {
        'hat1_d1': hat1_d1,
        'hat1_d2': hat1_d2,
        'tilde1_d1': tilde1_d1,
        'tilde1_d2': tilde1_d2,
        'hatY_d1': hatY_d1,
        'hatY_d2': hatY_d2,
        'dHY': dHY
    }


def compute_grr_tau(f1_hat, f2_hat, hat1_d1, hat1_d2, hatY_d1, hatY_d2):
    """
    Compute the GRR doubly robust estimator for treatment effect.
    
    The estimator is:
        tau_hat = mu1_hat - mu2_hat
    where:
        mu1_hat = mean(hatY_d1 - f1_hat * hat1_d1) + mean(f1_hat)
        mu2_hat = mean(hatY_d2 - f2_hat * hat1_d2) + mean(f2_hat)
    
    Args:
        f1_hat: predicted adjustment for d1 (n,)
        f2_hat: predicted adjustment for d2 (n,)
        hat1_d1, hat1_d2: IPW weights (n,)
        hatY_d1, hatY_d2: weighted outcomes (n,)
    
    Returns:
        tau_hat: estimated treatment effect
        mu1_hat: estimated mean for d1
        mu2_hat: estimated mean for d2
    """
    # mu_d1 = mean(hatY_d1 - f1_hat * hat1_d1) + mean(f1_hat)
    mu1_hat = np.mean(hatY_d1 - f1_hat * hat1_d1) + np.mean(f1_hat)
    
    # mu_d2 = mean(hatY_d2 - f2_hat * hat1_d2) + mean(f2_hat)
    mu2_hat = np.mean(hatY_d2 - f2_hat * hat1_d2) + np.mean(f2_hat)
    
    tau_hat = mu1_hat - mu2_hat
    
    return tau_hat, mu1_hat, mu2_hat


# =============================================================================
# Base GRR Class
# =============================================================================

class BaseGRR(ABC):
    """Abstract base class for GRR estimators."""
    
    @abstractmethod
    def fit(self, X, tilde1_d1, tilde1_d2, dHY, Kn):
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Return (f1_hat, f2_hat) predictions."""
        pass
    
    def compute_residual_loss(self, X, tilde1_d1, tilde1_d2, dHY, Kn):
        """Compute the weighted residual loss res' @ Kn @ res."""
        f1_hat, f2_hat = self.predict(X)
        res = dHY - f1_hat * tilde1_d1 + f2_hat * tilde1_d2
        return float(res @ Kn @ res)


# =============================================================================
# Linear GRR (Exact Solution)
# =============================================================================

class LinearGRR(BaseGRR):
    """
    Linear GRR estimator with exact weighted least squares solution.
    
    For f1(X) = X @ theta1 and f2(X) = X @ theta2, we minimize:
        (dHY - Z @ theta)' @ Kn @ (dHY - Z @ theta)
    where Z = [X * tilde1_d1, -X * tilde1_d2] and theta = [theta1; theta2].
    
    The solution is: theta = (Z' Kn Z + alpha*I)^{-1} Z' Kn dHY
    
    Supports: 'ols', 'ridge', 'lasso', 'elasticnet'
    """
    
    def __init__(self, reg_type='ridge', alpha=0.01, l1_ratio=0.5):
        """
        Args:
            reg_type: 'ols', 'ridge', 'lasso', or 'elasticnet'
            alpha: regularization strength (0 for OLS)
            l1_ratio: for elasticnet, ratio of L1 penalty (0=ridge, 1=lasso)
        """
        self.reg_type = reg_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.theta1_ = None
        self.theta2_ = None
        self.p_ = None
        
    def fit(self, X, tilde1_d1, tilde1_d2, dHY, Kn):
        """
        Fit the linear model.
        
        Args:
            X: covariates (n x p)
            tilde1_d1, tilde1_d2: demeaned weights (n,)
            dHY: target variable (n,)
            Kn: weight matrix (n x n)
        
        Returns:
            self
        """
        n, p = X.shape
        self.p_ = p
        
        # Build design matrix: Z = [X * tilde1_d1, -X * tilde1_d2]
        Z1 = X * tilde1_d1[:, None]  # (n, p)
        Z2 = X * tilde1_d2[:, None]  # (n, p)
        Z = np.hstack([Z1, -Z2])     # (n, 2p)
        
        if self.reg_type in ['ols', 'ridge']:
            # Direct solution: theta = (Z'KnZ + alpha*I)^{-1} Z'Kn y
            ZtKn = Z.T @ Kn           # (2p, n)
            ZtKnZ = ZtKn @ Z          # (2p, 2p)
            ZtKny = ZtKn @ dHY        # (2p,)
            
            if self.reg_type == 'ridge' and self.alpha > 0:
                ZtKnZ += self.alpha * np.eye(2 * p)
            
            # Use pseudo-inverse for numerical stability
            theta = np.linalg.lstsq(ZtKnZ, ZtKny, rcond=1e-2)[0]
            
        elif self.reg_type in ['lasso', 'elasticnet']:
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn is required for LASSO/ElasticNet")
            
            # Transform using Cholesky: Kn = L @ L.T
            # Then res'Kn res = ||L.T @ res||^2
            theta = self._fit_sparse(X, Z, tilde1_d1, tilde1_d2, dHY, Kn)
        else:
            raise ValueError(f"Unknown reg_type: {self.reg_type}")
        
        self.theta1_ = theta[:p]
        self.theta2_ = theta[p:]
        
        return self
    
    def _fit_sparse(self, X, Z, tilde1_d1, tilde1_d2, dHY, Kn):
        """
        Fit LASSO/ElasticNet using eigendecomposition-based transformation.
        
        Note: Kn is generally NOT positive semi-definite, so we cannot use
        Cholesky decomposition. Instead, we use eigendecomposition and handle
        negative eigenvalues by taking absolute values (which approximates
        the problem but allows standard solvers).
        """
        n = len(dHY)
        
        # Eigendecomposition: Kn = V @ diag(eigvals) @ V.T
        eigvals, eigvecs = np.linalg.eigh(Kn)
        
        # Handle non-PSD: take absolute values of eigenvalues
        # This is an approximation that preserves the "importance" structure
        # Alternatively, we could set negative eigenvalues to 0
        eigvals_abs = np.abs(eigvals)
        eigvals_abs = np.maximum(eigvals_abs, 1e-8)  # numerical stability
        
        # Construct transformation: L such that L.T @ L ≈ |Kn|
        # L = V @ diag(sqrt(|eigvals|))
        L = eigvecs @ np.diag(np.sqrt(eigvals_abs))
        
        # Transform: Z_new = L.T @ Z, y_new = L.T @ dHY
        Z_new = L.T @ Z
        y_new = L.T @ dHY
        
        if self.reg_type == 'lasso':
            model = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000)
        else:
            model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                              fit_intercept=False, max_iter=10000)
        
        model.fit(Z_new, y_new)
        return model.coef_
    
    def predict(self, X):
        """
        Predict f1 and f2.
        
        Args:
            X: covariates (n x p)
        
        Returns:
            f1_hat, f2_hat: predictions (n,), (n,)
        """
        if self.theta1_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        f1_hat = X @ self.theta1_
        f2_hat = X @ self.theta2_
        return f1_hat, f2_hat


# =============================================================================
# Sklearn-based GRR (Approximate via transformed problem)
# =============================================================================

class SklearnGRR(BaseGRR):
    """
    GRR using sklearn regressors.
    
    For non-linear models (RF, GBM, etc.), we have two approaches:
    
    1. Eigendecomposition transformation (use_transform=True):
       Transform the problem using eigendecomposition of Kn.
       Note: Kn is generally NOT positive semi-definite, so we use
       |eigenvalues| as an approximation. This mixes samples.
    
    2. Diagonal approximation (use_transform=False, recommended for trees):
       Use row sums of Kn as sample weights. This is simpler and often
       works well for tree-based models that support sample_weight.
    
    For tree-based models, the diagonal approximation is recommended.
    """
    
    def __init__(self, base_estimator=None, use_transform=False, 
                 separate_models=True):
        """
        Args:
            base_estimator: sklearn regressor (default: Ridge)
            use_transform: if True, use eigendecomposition transformation
                          if False, use diagonal approximation (sample weights)
                          Default False (recommended for tree-based models)
            separate_models: if True, fit separate models for f1 and f2
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for SklearnGRR")
        
        if base_estimator is None:
            base_estimator = Ridge(alpha=1.0)
        
        self.base_estimator = base_estimator
        self.use_transform = use_transform
        self.separate_models = separate_models
        self.model1_ = None
        self.model2_ = None
        self.joint_model_ = None
        
    def fit(self, X, tilde1_d1, tilde1_d2, dHY, Kn):
        """Fit the model."""
        n, p = X.shape
        
        if self.use_transform:
            return self._fit_transform(X, tilde1_d1, tilde1_d2, dHY, Kn)
        else:
            return self._fit_weighted(X, tilde1_d1, tilde1_d2, dHY, Kn)
    
    def _fit_transform(self, X, tilde1_d1, tilde1_d2, dHY, Kn):
        """
        Fit using eigendecomposition-based transformation.
        
        Note: Kn is generally NOT positive semi-definite. We handle this by
        using eigendecomposition and taking absolute values of eigenvalues.
        This is an approximation that preserves the importance structure.
        """
        n, p = X.shape
        
        # Eigendecomposition (works for any symmetric matrix)
        eigvals, eigvecs = np.linalg.eigh(Kn)
        
        # Handle non-PSD: take absolute values of eigenvalues
        eigvals_abs = np.abs(eigvals)
        eigvals_abs = np.maximum(eigvals_abs, 1e-8)
        
        # Construct transformation matrix
        L = eigvecs @ np.diag(np.sqrt(eigvals_abs))
        
        # Transform target
        y_transformed = L.T @ dHY
        
        if self.separate_models:
            # Fit two separate models for interpretability
            # This is an approximation but often works well
            
            # For f1: regress L.T @ (dHY on units with D=d1) ~ L.T @ X
            # We use the full transformed problem with appropriate weighting
            
            # Model 1: predict tilde1_d1 * f1(X) contribution
            Z1 = X * tilde1_d1[:, None]
            Z1_transformed = L.T @ Z1
            
            self.model1_ = clone(self.base_estimator)
            self.model1_.fit(Z1_transformed, y_transformed)
            
            # Residual for model 2
            pred1 = self.model1_.predict(Z1_transformed)
            residual = y_transformed - pred1
            
            # Model 2: predict -tilde1_d2 * f2(X) contribution
            Z2 = X * tilde1_d2[:, None]
            Z2_transformed = L.T @ (-Z2)
            
            self.model2_ = clone(self.base_estimator)
            self.model2_.fit(Z2_transformed, residual)
            
        else:
            # Joint model: stack features
            Z1 = X * tilde1_d1[:, None]
            Z2 = X * tilde1_d2[:, None]
            Z = np.hstack([Z1, -Z2])
            Z_transformed = L.T @ Z
            
            self.joint_model_ = clone(self.base_estimator)
            self.joint_model_.fit(Z_transformed, y_transformed)
        
        self._L = L  # Store for prediction
        return self
    
    def _fit_weighted(self, X, tilde1_d1, tilde1_d2, dHY, Kn):
        """Fit using diagonal weight approximation."""
        n, p = X.shape
        
        # Use row sums as sample weights (approximation)
        sample_weight = np.maximum(Kn.sum(axis=1), 1e-6)
        
        # Build stacked design
        Z1 = X * tilde1_d1[:, None]
        Z2 = X * tilde1_d2[:, None]
        Z = np.hstack([Z1, -Z2])
        
        self.joint_model_ = clone(self.base_estimator)
        
        # Try to use sample_weight if supported
        try:
            self.joint_model_.fit(Z, dHY, sample_weight=sample_weight)
        except TypeError:
            # Model doesn't support sample_weight
            self.joint_model_.fit(Z, dHY)
        
        return self
    
    def predict(self, X):
        """Predict f1 and f2."""
        n, p = X.shape
        
        if self.separate_models and self.model1_ is not None:
            # For separate models, we need to extract predictions carefully
            # The models were trained on transformed space, so we approximate
            
            # Simple approach: use the fitted coefficients if linear
            # For non-linear, we need a different strategy
            
            # Predict on original scale (approximation)
            f1_hat = self.model1_.predict(X)
            f2_hat = -self.model2_.predict(X)  # Note the negative
            
        elif self.joint_model_ is not None:
            # For joint model, extract coefficients if linear
            if hasattr(self.joint_model_, 'coef_'):
                coef = self.joint_model_.coef_
                theta1 = coef[:p]
                theta2 = coef[p:]
                f1_hat = X @ theta1
                f2_hat = X @ (-theta2)  # Note: Z had -Z2
            else:
                # Non-linear model: predict with indicator features
                # This is trickier - we use a simple approximation
                Z1 = X * np.ones(n)[:, None]  # Assume tilde1_d1 = 1
                Z2 = X * np.zeros(n)[:, None]
                Z = np.hstack([Z1, -Z2])
                f1_hat = self.joint_model_.predict(Z)
                
                Z1 = X * np.zeros(n)[:, None]
                Z2 = X * np.ones(n)[:, None]
                Z = np.hstack([Z1, -Z2])
                f2_hat = -self.joint_model_.predict(Z)
        else:
            raise ValueError("Model not fitted.")
        
        return f1_hat, f2_hat


# =============================================================================
# PyTorch-based GRR (Neural Networks)
# =============================================================================

if TORCH_AVAILABLE:
    class TorchMLP(nn.Module):
        """MLP module for f1 or f2."""
        
        def __init__(self, input_dim, hidden_dims=[64, 32], 
                     activation='relu', dropout=0.0):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            act_fn = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'gelu': nn.GELU,
                'leaky_relu': nn.LeakyReLU
            }.get(activation, nn.ReLU)
            
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x).squeeze(-1)


    class TorchGRR(BaseGRR, nn.Module):
        """
        PyTorch-based GRR for neural network function classes.
        
        Minimizes res' @ Kn @ res using gradient descent.
        """
        
        def __init__(self, input_dim, model_type='mlp', 
                     hidden_dims=[64, 32], activation='relu', dropout=0.0,
                     share_architecture=False):
            """
            Args:
                input_dim: number of input features
                model_type: 'linear' or 'mlp'
                hidden_dims: hidden layer sizes for MLP
                activation: activation function
                dropout: dropout rate
                share_architecture: if True, f1 and f2 share weights (not recommended)
            """
            nn.Module.__init__(self)
            
            self.input_dim = input_dim
            self.model_type = model_type
            self.share_architecture = share_architecture
            
            if model_type == 'linear':
                self.f1 = nn.Linear(input_dim, 1, bias=True)
                self.f2 = nn.Linear(input_dim, 1, bias=True)
            elif model_type == 'mlp':
                self.f1 = TorchMLP(input_dim, hidden_dims, activation, dropout)
                if share_architecture:
                    self.f2 = self.f1
                else:
                    self.f2 = TorchMLP(input_dim, hidden_dims, activation, dropout)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            self._fitted = False
        
        def forward(self, X):
            """Forward pass returning (f1, f2) predictions."""
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            
            f1_out = self.f1(X)
            f2_out = self.f2(X)
            
            if f1_out.dim() > 1:
                f1_out = f1_out.squeeze(-1)
            if f2_out.dim() > 1:
                f2_out = f2_out.squeeze(-1)
            
            return f1_out, f2_out
        
        def compute_loss(self, X, tilde1_d1, tilde1_d2, dHY, Kn):
            """
            Compute weighted L2 loss: res' @ Kn @ res
            """
            f1, f2 = self(X)
            
            # res = dHY - f1 * tilde1_d1 + f2 * tilde1_d2
            res = dHY - f1 * tilde1_d1 + f2 * tilde1_d2
            
            # loss = res' @ Kn @ res
            Kn_res = torch.mv(Kn, res)  # Kn @ res
            loss = torch.dot(res, Kn_res)
            
            return loss
        
        def fit(self, X, tilde1_d1, tilde1_d2, dHY, Kn,
                lr=0.001, n_epochs=1000, weight_decay=0.0,
                early_stopping=True, patience=50, min_delta=1e-6,
                verbose=False, batch_size=None):
            """
            Fit using gradient descent.
            
            Args:
                X: covariates (n x p)
                tilde1_d1, tilde1_d2: demeaned weights (n,)
                dHY: target (n,)
                Kn: weight matrix (n x n)
                lr: learning rate
                n_epochs: max epochs
                weight_decay: L2 regularization
                early_stopping: stop if loss plateaus
                patience: epochs to wait before stopping
                min_delta: minimum improvement threshold
                verbose: print progress
                batch_size: if None, use full batch
            
            Returns:
                self
            """
            # Convert to tensors
            X_t = torch.tensor(X, dtype=torch.float32)
            tilde1_d1_t = torch.tensor(tilde1_d1, dtype=torch.float32)
            tilde1_d2_t = torch.tensor(tilde1_d2, dtype=torch.float32)
            dHY_t = torch.tensor(dHY, dtype=torch.float32)
            Kn_t = torch.tensor(Kn, dtype=torch.float32)
            
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, 
                                         weight_decay=weight_decay)
            
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                loss = self.compute_loss(X_t, tilde1_d1_t, tilde1_d2_t, 
                                        dHY_t, Kn_t)
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()
                
                # Early stopping
                if early_stopping:
                    if loss_val < best_loss - min_delta:
                        best_loss = loss_val
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss_val:.6f}")
            
            self._fitted = True
            return self
        
        def predict(self, X):
            """Predict f1 and f2."""
            if not self._fitted:
                raise ValueError("Model not fitted. Call fit() first.")
            
            self.eval()
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_t = torch.tensor(X, dtype=torch.float32)
                else:
                    X_t = X
                f1, f2 = self(X_t)
                
                if isinstance(f1, torch.Tensor):
                    f1 = f1.numpy()
                    f2 = f2.numpy()
            
            return f1, f2


# =============================================================================
# Main GRR Interface
# =============================================================================

def fit_grr(X, tilde1_d1, tilde1_d2, dHY, Kn,
            method='linear', **kwargs):
    """
    Fit a GRR model.
    
    Args:
        X: covariates (n x p)
        tilde1_d1, tilde1_d2: demeaned weights (n,)
        dHY: target (n,)
        Kn: weight matrix (n x n)
        method: one of:
            - 'linear': LinearGRR with OLS
            - 'ridge': LinearGRR with Ridge penalty
            - 'lasso': LinearGRR with LASSO penalty
            - 'elasticnet': LinearGRR with ElasticNet
            - 'rf': Random Forest via SklearnGRR
            - 'gbm': Gradient Boosting via SklearnGRR
            - 'mlp': Neural network via TorchGRR
        **kwargs: method-specific arguments
    
    Returns:
        Fitted GRR model
    """
    method = method.lower()
    
    if method == 'linear':
        model = LinearGRR(reg_type='ols')
    elif method == 'ridge':
        model = LinearGRR(reg_type='ridge', alpha=kwargs.get('alpha', 0.01))
    elif method == 'lasso':
        model = LinearGRR(reg_type='lasso', alpha=kwargs.get('alpha', 0.01))
    elif method == 'elasticnet':
        model = LinearGRR(reg_type='elasticnet', 
                         alpha=kwargs.get('alpha', 0.01),
                         l1_ratio=kwargs.get('l1_ratio', 0.5))
    elif method == 'rf':
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for Random Forest")
        base = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_leaf=kwargs.get('min_samples_leaf', 5),
            random_state=kwargs.get('random_state', None)
        )
        # use_transform=False by default for trees (diagonal approximation)
        model = SklearnGRR(base_estimator=base, 
                          use_transform=kwargs.get('use_transform', False),
                          separate_models=False)
    elif method == 'gbm':
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for GBM")
        base = GradientBoostingRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 3),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=kwargs.get('random_state', None)
        )
        # use_transform=False by default for trees (diagonal approximation)
        model = SklearnGRR(base_estimator=base,
                          use_transform=kwargs.get('use_transform', False),
                          separate_models=False)
    elif method == 'mlp':
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MLP")
        model = TorchGRR(
            input_dim=X.shape[1],
            model_type='mlp',
            hidden_dims=kwargs.get('hidden_dims', [64, 32]),
            activation=kwargs.get('activation', 'relu'),
            dropout=kwargs.get('dropout', 0.0)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit the model
    if method == 'mlp':
        model.fit(X, tilde1_d1, tilde1_d2, dHY, Kn,
                 lr=kwargs.get('lr', 0.001),
                 n_epochs=kwargs.get('n_epochs', 1000),
                 weight_decay=kwargs.get('weight_decay', 0.0),
                 early_stopping=kwargs.get('early_stopping', True),
                 patience=kwargs.get('patience', 50),
                 verbose=kwargs.get('verbose', False))
    else:
        model.fit(X, tilde1_d1, tilde1_d2, dHY, Kn)
    
    return model


def grr_estimator(Y, D, X, d1, d2, pi_id_all, Kn,
                  method='ridge', **kwargs):
    """
    Compute the GRR treatment effect estimator.
    
    This is the main interface for GRR estimation. It:
    1. Prepares inputs (hat1, tilde1, hatY, dHY)
    2. Fits the chosen model to minimize res' @ Kn @ res
    3. Computes the doubly robust estimator tau_hat
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        X: covariates (n x p)
        d1, d2: exposure levels to compare
        pi_id_all: propensity scores (n x k)
        Kn: weight matrix (n x n), typically from build_kn_matrix
        method: model type ('linear', 'ridge', 'lasso', 'elasticnet',
                           'rf', 'gbm', 'mlp')
        **kwargs: method-specific arguments
    
    Returns:
        dict with:
            tau_hat: estimated treatment effect
            mu1_hat: estimated mean for d1
            mu2_hat: estimated mean for d2
            f1_hat: predictions for adjustment function 1 (n,)
            f2_hat: predictions for adjustment function 2 (n,)
            model: fitted model object
            loss: final weighted loss value
    """
    # Prepare inputs
    inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all)
    
    # Fit model
    model = fit_grr(
        X, 
        inputs['tilde1_d1'], 
        inputs['tilde1_d2'], 
        inputs['dHY'], 
        Kn,
        method=method,
        **kwargs
    )
    
    # Get predictions
    f1_hat, f2_hat = model.predict(X)
    
    # Compute treatment effect
    tau_hat, mu1_hat, mu2_hat = compute_grr_tau(
        f1_hat, f2_hat,
        inputs['hat1_d1'], inputs['hat1_d2'],
        inputs['hatY_d1'], inputs['hatY_d2']
    )
    
    # Compute loss
    loss = model.compute_residual_loss(
        X, inputs['tilde1_d1'], inputs['tilde1_d2'], 
        inputs['dHY'], Kn
    )
    
    return {
        'tau_hat': tau_hat,
        'mu1_hat': mu1_hat,
        'mu2_hat': mu2_hat,
        'f1_hat': f1_hat,
        'f2_hat': f2_hat,
        'model': model,
        'loss': loss
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def compare_grr_methods(Y, D, X, d1, d2, pi_id_all, Kn,
                        methods=['linear', 'ridge', 'lasso'],
                        verbose=True, **kwargs):
    """
    Compare multiple GRR methods on the same data.
    
    Args:
        Y, D, X, d1, d2, pi_id_all, Kn: standard GRR inputs
        methods: list of methods to compare
        verbose: print results
        **kwargs: passed to all methods
    
    Returns:
        dict mapping method -> results
    """
    results = {}
    
    for method in methods:
        try:
            result = grr_estimator(Y, D, X, d1, d2, pi_id_all, Kn,
                                  method=method, **kwargs)
            results[method] = result
            
            if verbose:
                print(f"{method:12s}: tau_hat = {result['tau_hat']:.6f}, "
                      f"loss = {result['loss']:.2f}")
        except Exception as e:
            if verbose:
                print(f"{method:12s}: FAILED - {str(e)}")
            results[method] = None
    
    return results

