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

def prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=None, X2=None, eps=1e-12):
    """
    Prepare inputs for GRR estimation.
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        X: covariate matrix (n x p) - used if X1, X2 not provided
        d1, d2: exposure levels to compare
        pi_id_all: propensity score matrix (n x k)
        X1: covariates for level d1 (n x p1), optional
        X2: covariates for level d2 (n x p2), optional
        eps: small constant for numerical stability
    
    Returns:
        dict with keys:
            hat1_d1, hat1_d2: IPW weights (n,)
            tilde1_d1, tilde1_d2: demeaned weights (n,)
            hatY_d1, hatY_d2: weighted outcomes (n,)
            dHY: difference in weighted outcomes (n,)
            X1, X2: covariates for each level
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
    
    # Handle covariates: use X1, X2 if provided, otherwise use X for both
    if X1 is None:
        X1 = X
    if X2 is None:
        X2 = X
    
    return {
        'hat1_d1': hat1_d1,
        'hat1_d2': hat1_d2,
        'tilde1_d1': tilde1_d1,
        'tilde1_d2': tilde1_d2,
        'hatY_d1': hatY_d1,
        'hatY_d2': hatY_d2,
        'dHY': dHY,
        'X1': X1,
        'X2': X2
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
    def fit(self, X1, X2, tilde1_d1, tilde1_d2, dHY, Kn):
        """
        Fit the model.
        
        Args:
            X1: covariates for level d1 (n x p1)
            X2: covariates for level d2 (n x p2)
            tilde1_d1, tilde1_d2: demeaned weights (n,)
            dHY: target variable (n,)
            Kn: weight matrix (n x n)
        """
        pass
    
    @abstractmethod
    def predict(self, X1, X2):
        """
        Return (f1_hat, f2_hat) predictions.
        
        Args:
            X1: covariates for level d1 (n x p1)
            X2: covariates for level d2 (n x p2)
        
        Returns:
            f1_hat, f2_hat: predictions (n,), (n,)
        """
        pass
    
    def compute_residual_loss(self, X1, X2, tilde1_d1, tilde1_d2, dHY, Kn):
        """Compute the weighted residual loss res' @ Kn @ res."""
        f1_hat, f2_hat = self.predict(X1, X2)
        res = dHY - f1_hat * tilde1_d1 + f2_hat * tilde1_d2
        return float(res @ Kn @ res)


# =============================================================================
# Linear GRR (Exact Solution)
# =============================================================================

class LinearGRR(BaseGRR):
    """
    Linear GRR estimator with exact weighted least squares solution.
    
    For f1(X1) = X1 @ theta1 and f2(X2) = X2 @ theta2, we minimize:
        (dHY - Z @ theta)' @ Kn @ (dHY - Z @ theta)
    where Z = [X1 * tilde1_d1, -X2 * tilde1_d2] and theta = [theta1; theta2].
    
    The solution is: theta = (Z' Kn Z + alpha*I)^{-1} Z' Kn dHY
    
    Supports: 'ols', 'ridge', 'lasso', 'elasticnet'
    
    Note: X1 and X2 can have different dimensions (p1 and p2).
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
        self.p1_ = None
        self.p2_ = None
        
    def fit(self, X1, X2, tilde1_d1, tilde1_d2, dHY, Kn):
        """
        Fit the linear model.
        
        Args:
            X1: covariates for d1 (n x p1)
            X2: covariates for d2 (n x p2)
            tilde1_d1, tilde1_d2: demeaned weights (n,)
            dHY: target variable (n,)
            Kn: weight matrix (n x n)
        
        Returns:
            self
        """
        n = len(dHY)
        p1 = X1.shape[1]
        p2 = X2.shape[1]
        self.p1_ = p1
        self.p2_ = p2
        
        # Build design matrix: Z = [X1 * tilde1_d1, -X2 * tilde1_d2]
        Z1 = X1 * tilde1_d1[:, None]  # (n, p1)
        Z2 = X2 * tilde1_d2[:, None]  # (n, p2)
        Z = np.hstack([Z1, -Z2])       # (n, p1+p2)
        
        if self.reg_type in ['ols', 'ridge']:
            # Direct solution: theta = (Z'KnZ + alpha*I)^{-1} Z'Kn y
            ZtKn = Z.T @ Kn           # (p1+p2, n)
            ZtKnZ = ZtKn @ Z          # (p1+p2, p1+p2)
            ZtKny = ZtKn @ dHY        # (p1+p2,)
            
            if self.reg_type == 'ridge' and self.alpha > 0:
                ZtKnZ += self.alpha * np.eye(p1 + p2)
            
            # Use pseudo-inverse for numerical stability
            theta = np.linalg.lstsq(ZtKnZ, ZtKny, rcond=1e-2)[0]
            
        elif self.reg_type in ['lasso', 'elasticnet']:
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn is required for LASSO/ElasticNet")
            
            # Transform using eigendecomposition
            theta = self._fit_sparse(Z, dHY, Kn)
        else:
            raise ValueError(f"Unknown reg_type: {self.reg_type}")
        
        self.theta1_ = theta[:p1]
        self.theta2_ = theta[p1:]
        
        return self
    
    def _fit_sparse(self, Z, dHY, Kn):
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
        eigvals_abs = np.abs(eigvals)
        eigvals_abs = np.maximum(eigvals_abs, 1e-8)  # numerical stability
        
        # Construct transformation: L such that L.T @ L ≈ |Kn|
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
    
    def predict(self, X1, X2):
        """
        Predict f1 and f2.
        
        Args:
            X1: covariates for d1 (n x p1)
            X2: covariates for d2 (n x p2)
        
        Returns:
            f1_hat, f2_hat: predictions (n,), (n,)
        """
        if self.theta1_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        f1_hat = X1 @ self.theta1_
        f2_hat = X2 @ self.theta2_
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
    
    Note: X1 and X2 can have different dimensions (p1 and p2).
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
        self.p1_ = None
        self.p2_ = None
        
    def fit(self, X1, X2, tilde1_d1, tilde1_d2, dHY, Kn):
        """Fit the model."""
        n = len(dHY)
        self.p1_ = X1.shape[1]
        self.p2_ = X2.shape[1]
        
        if self.use_transform:
            return self._fit_transform(X1, X2, tilde1_d1, tilde1_d2, dHY, Kn)
        else:
            return self._fit_weighted(X1, X2, tilde1_d1, tilde1_d2, dHY, Kn)
    
    def _fit_transform(self, X1, X2, tilde1_d1, tilde1_d2, dHY, Kn):
        """
        Fit using eigendecomposition-based transformation.
        
        Note: Kn is generally NOT positive semi-definite. We handle this by
        using eigendecomposition and taking absolute values of eigenvalues.
        This is an approximation that preserves the importance structure.
        """
        n = len(dHY)
        
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
            
            # Model 1: predict tilde1_d1 * f1(X1) contribution
            Z1 = X1 * tilde1_d1[:, None]
            Z1_transformed = L.T @ Z1
            
            self.model1_ = clone(self.base_estimator)
            self.model1_.fit(Z1_transformed, y_transformed)
            
            # Residual for model 2
            pred1 = self.model1_.predict(Z1_transformed)
            residual = y_transformed - pred1
            
            # Model 2: predict -tilde1_d2 * f2(X2) contribution
            Z2 = X2 * tilde1_d2[:, None]
            Z2_transformed = L.T @ (-Z2)
            
            self.model2_ = clone(self.base_estimator)
            self.model2_.fit(Z2_transformed, residual)
            
        else:
            # Joint model: stack features
            Z1 = X1 * tilde1_d1[:, None]
            Z2 = X2 * tilde1_d2[:, None]
            Z = np.hstack([Z1, -Z2])
            Z_transformed = L.T @ Z
            
            self.joint_model_ = clone(self.base_estimator)
            self.joint_model_.fit(Z_transformed, y_transformed)
        
        self._L = L  # Store for prediction
        return self
    
    def _fit_weighted(self, X1, X2, tilde1_d1, tilde1_d2, dHY, Kn):
        """Fit using diagonal weight approximation."""
        n = len(dHY)
        
        # Use row sums as sample weights (approximation)
        sample_weight = np.maximum(Kn.sum(axis=1), 1e-6)
        
        # Build stacked design
        Z1 = X1 * tilde1_d1[:, None]
        Z2 = X2 * tilde1_d2[:, None]
        Z = np.hstack([Z1, -Z2])
        
        self.joint_model_ = clone(self.base_estimator)
        
        # Try to use sample_weight if supported
        try:
            self.joint_model_.fit(Z, dHY, sample_weight=sample_weight)
        except TypeError:
            # Model doesn't support sample_weight
            self.joint_model_.fit(Z, dHY)
        
        return self
    
    def predict(self, X1, X2):
        """Predict f1 and f2."""
        n = X1.shape[0]
        p1 = self.p1_
        p2 = self.p2_
        
        if self.separate_models and self.model1_ is not None:
            # For separate models, predict on original scale (approximation)
            f1_hat = self.model1_.predict(X1)
            f2_hat = -self.model2_.predict(X2)  # Note the negative
            
        elif self.joint_model_ is not None:
            # For joint model, extract coefficients if linear
            if hasattr(self.joint_model_, 'coef_'):
                coef = self.joint_model_.coef_
                theta1 = coef[:p1]
                theta2 = coef[p1:]
                f1_hat = X1 @ theta1
                f2_hat = X2 @ (-theta2)  # Note: Z had -Z2
            else:
                # Non-linear model: predict with indicator features
                Z1 = X1 * np.ones(n)[:, None]  # Assume tilde1_d1 = 1
                Z2 = X2 * np.zeros(n)[:, None]
                Z = np.hstack([Z1, -Z2])
                f1_hat = self.joint_model_.predict(Z)
                
                Z1 = X1 * np.zeros(n)[:, None]
                Z2 = X2 * np.ones(n)[:, None]
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
        Automatically uses GPU if available.
        
        Note: X1 and X2 can have different dimensions (p1 and p2).
        """
        
        def __init__(self, input_dim1, input_dim2=None, model_type='mlp', 
                     hidden_dims=[64, 32], activation='relu', dropout=0.0,
                     share_architecture=False, device=None):
            """
            Args:
                input_dim1: number of input features for f1
                input_dim2: number of input features for f2 (default: same as input_dim1)
                model_type: 'linear' or 'mlp'
                hidden_dims: hidden layer sizes for MLP
                activation: activation function
                dropout: dropout rate
                share_architecture: if True, f1 and f2 share weights (requires same input dims)
                device: 'cuda', 'cpu', or None (auto-detect)
            """
            nn.Module.__init__(self)
            
            # Auto-detect device (prefer CUDA)
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            # Print device info once
            if not hasattr(TorchGRR, '_device_printed'):
                print(f"[TorchGRR] Using device: {self.device}")
                if self.device.type == 'cuda':
                    print(f"[TorchGRR] GPU: {torch.cuda.get_device_name(0)}")
                TorchGRR._device_printed = True
            
            if input_dim2 is None:
                input_dim2 = input_dim1
            
            self.input_dim1 = input_dim1
            self.input_dim2 = input_dim2
            self.model_type = model_type
            self.share_architecture = share_architecture
            
            if share_architecture and input_dim1 != input_dim2:
                raise ValueError("Cannot share architecture when input dimensions differ")
            
            if model_type == 'linear':
                self.f1 = nn.Linear(input_dim1, 1, bias=True)
                self.f2 = nn.Linear(input_dim2, 1, bias=True)
            elif model_type == 'mlp':
                self.f1 = TorchMLP(input_dim1, hidden_dims, activation, dropout)
                if share_architecture:
                    self.f2 = self.f1
                else:
                    self.f2 = TorchMLP(input_dim2, hidden_dims, activation, dropout)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            # Move model to device
            self.to(self.device)
            
            self._fitted = False
        
        def forward(self, X1, X2):
            """Forward pass returning (f1, f2) predictions."""
            if isinstance(X1, np.ndarray):
                X1 = torch.tensor(X1, dtype=torch.float32, device=self.device)
            elif X1.device != self.device:
                X1 = X1.to(self.device)
            
            if isinstance(X2, np.ndarray):
                X2 = torch.tensor(X2, dtype=torch.float32, device=self.device)
            elif X2.device != self.device:
                X2 = X2.to(self.device)
            
            f1_out = self.f1(X1)
            f2_out = self.f2(X2)
            
            if f1_out.dim() > 1:
                f1_out = f1_out.squeeze(-1)
            if f2_out.dim() > 1:
                f2_out = f2_out.squeeze(-1)
            
            return f1_out, f2_out
        
        def compute_loss(self, X1, X2, tilde1_d1, tilde1_d2, dHY, Kn):
            """
            Compute weighted L2 loss: res' @ Kn @ res
            """
            f1, f2 = self(X1, X2)
            
            # res = dHY - f1 * tilde1_d1 + f2 * tilde1_d2
            res = dHY - f1 * tilde1_d1 + f2 * tilde1_d2
            
            # loss = res' @ Kn @ res
            Kn_res = torch.mv(Kn, res)  # Kn @ res
            loss = torch.dot(res, Kn_res)
            
            return loss
        
        def fit(self, X1, X2, tilde1_d1, tilde1_d2, dHY, Kn,
                lr=0.001, n_epochs=1000, weight_decay=0.0,
                early_stopping=True, patience=50, min_delta=1e-6,
                verbose=False, batch_size=None):
            """
            Fit using gradient descent.
            
            Args:
                X1: covariates for d1 (n x p1)
                X2: covariates for d2 (n x p2)
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
            # Convert to tensors and move to device
            X1_t = torch.tensor(X1, dtype=torch.float32, device=self.device)
            X2_t = torch.tensor(X2, dtype=torch.float32, device=self.device)
            tilde1_d1_t = torch.tensor(tilde1_d1, dtype=torch.float32, device=self.device)
            tilde1_d2_t = torch.tensor(tilde1_d2, dtype=torch.float32, device=self.device)
            dHY_t = torch.tensor(dHY, dtype=torch.float32, device=self.device)
            Kn_t = torch.tensor(Kn, dtype=torch.float32, device=self.device)
            
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, 
                                         weight_decay=weight_decay)
            
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                loss = self.compute_loss(X1_t, X2_t, tilde1_d1_t, tilde1_d2_t, 
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
        
        def predict(self, X1, X2):
            """Predict f1 and f2."""
            if not self._fitted:
                raise ValueError("Model not fitted. Call fit() first.")
            
            self.eval()
            with torch.no_grad():
                if isinstance(X1, np.ndarray):
                    X1_t = torch.tensor(X1, dtype=torch.float32, device=self.device)
                else:
                    X1_t = X1.to(self.device)
                
                if isinstance(X2, np.ndarray):
                    X2_t = torch.tensor(X2, dtype=torch.float32, device=self.device)
                else:
                    X2_t = X2.to(self.device)
                
                f1, f2 = self(X1_t, X2_t)
                
                if isinstance(f1, torch.Tensor):
                    f1 = f1.cpu().numpy()
                    f2 = f2.cpu().numpy()
            
            return f1, f2


# =============================================================================
# Main GRR Interface
# =============================================================================

def fit_grr(X1, X2, tilde1_d1, tilde1_d2, dHY, Kn,
            method='linear', **kwargs):
    """
    Fit a GRR model.
    
    Args:
        X1: covariates for d1 (n x p1)
        X2: covariates for d2 (n x p2)
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
            input_dim1=X1.shape[1],
            input_dim2=X2.shape[1],
            model_type='mlp',
            hidden_dims=kwargs.get('hidden_dims', [64, 32]),
            activation=kwargs.get('activation', 'relu'),
            dropout=kwargs.get('dropout', 0.0)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit the model
    if method == 'mlp':
        model.fit(X1, X2, tilde1_d1, tilde1_d2, dHY, Kn,
                 lr=kwargs.get('lr', 0.001),
                 n_epochs=kwargs.get('n_epochs', 1000),
                 weight_decay=kwargs.get('weight_decay', 0.0),
                 early_stopping=kwargs.get('early_stopping', True),
                 patience=kwargs.get('patience', 50),
                 verbose=kwargs.get('verbose', False))
    else:
        model.fit(X1, X2, tilde1_d1, tilde1_d2, dHY, Kn)
    
    return model


def grr_estimator(Y, D, X, d1, d2, pi_id_all, Kn,
                  method='ridge', X1=None, X2=None, **kwargs):
    """
    Compute the GRR treatment effect estimator.
    
    This is the main interface for GRR estimation. It:
    1. Prepares inputs (hat1, tilde1, hatY, dHY)
    2. Fits the chosen model to minimize res' @ Kn @ res
    3. Computes the doubly robust estimator tau_hat
    
    Args:
        Y: observed outcomes (n,)
        D: observed exposure (n,)
        X: covariates (n x p) - used if X1, X2 not provided
        d1, d2: exposure levels to compare
        pi_id_all: propensity scores (n x k)
        Kn: weight matrix (n x n), typically from build_kn_matrix
        method: model type ('linear', 'ridge', 'lasso', 'elasticnet',
                           'rf', 'gbm', 'mlp')
        X1: covariates for level d1 (n x p1), optional
        X2: covariates for level d2 (n x p2), optional
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
    # Prepare inputs (with optional X1, X2)
    inputs = prepare_grr_inputs(Y, D, X, d1, d2, pi_id_all, X1=X1, X2=X2)
    
    # Get the covariates to use
    X1_use = inputs['X1']
    X2_use = inputs['X2']
    
    # Fit model
    model = fit_grr(
        X1_use,
        X2_use,
        inputs['tilde1_d1'], 
        inputs['tilde1_d2'], 
        inputs['dHY'], 
        Kn,
        method=method,
        **kwargs
    )
    
    # Get predictions
    f1_hat, f2_hat = model.predict(X1_use, X2_use)
    
    # Compute treatment effect
    tau_hat, mu1_hat, mu2_hat = compute_grr_tau(
        f1_hat, f2_hat,
        inputs['hat1_d1'], inputs['hat1_d2'],
        inputs['hatY_d1'], inputs['hatY_d2']
    )
    
    # Compute loss
    loss = model.compute_residual_loss(
        X1_use, X2_use, inputs['tilde1_d1'], inputs['tilde1_d2'], 
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
                        X1=None, X2=None, verbose=True, **kwargs):
    """
    Compare multiple GRR methods on the same data.
    
    Args:
        Y, D, X, d1, d2, pi_id_all, Kn: standard GRR inputs
        methods: list of methods to compare
        X1: covariates for level d1 (n x p1), optional
        X2: covariates for level d2 (n x p2), optional
        verbose: print results
        **kwargs: passed to all methods
    
    Returns:
        dict mapping method -> results
    """
    results = {}
    
    for method in methods:
        try:
            result = grr_estimator(Y, D, X, d1, d2, pi_id_all, Kn,
                                  method=method, X1=X1, X2=X2, **kwargs)
            results[method] = result
            
            if verbose:
                print(f"{method:12s}: tau_hat = {result['tau_hat']:.6f}, "
                      f"loss = {result['loss']:.2f}")
        except Exception as e:
            if verbose:
                print(f"{method:12s}: FAILED - {str(e)}")
            results[method] = None
    
    return results


# =============================================================================
# GAT Embedding Function
# =============================================================================

def GAT_embedding(X, A, D, Y, k, d1=None, d2=None,
                  gat_hidden_dims=[64, 32],
                  rep_dim=16,
                  head_hidden_dims=[32, 16],
                  heads=4,
                  dropout=0.1,
                  lambda_ipm=0.0,
                  n_epochs=300,
                  lr=0.001,
                  weight_decay=1e-4,
                  early_stopping=True,
                  patience=50,
                  verbose=True,
                  return_model=False):
    """
    Train a GAT model and extract embeddings for GRR estimation.
    
    This function trains a Graph Attention Network using the GAT-embedder module
    and returns two sets of embeddings:
    
    1. X_GAT_root: Shared backbone embeddings from the GAT encoder (last attention 
       layer output, before the outcome heads). Shape: (n, rep_dim)
       
    2. X_GAT_leaf: Per-exposure-level embeddings from the last hidden layer of 
       each outcome head MLP. Returns a list of k arrays, each of shape 
       (n, last_hidden_dim), or if d1/d2 are specified, returns just 
       (X1_leaf, X2_leaf) for the two specified exposure levels.
    
    Args:
        X: node covariates (n x p), numpy array
        A: adjacency matrix (n x n), numpy array
        D: exposure levels (n,), numpy array with values in {0, ..., k-1}
        Y: observed outcomes (n,), numpy array
        k: number of exposure levels
        d1: first exposure level for comparison (optional)
        d2: second exposure level for comparison (optional)
        gat_hidden_dims: hidden dimensions for GAT encoder
        rep_dim: dimension of GAT output representations
        head_hidden_dims: hidden dimensions for outcome MLP heads
        heads: number of attention heads in GAT
        dropout: dropout rate
        lambda_ipm: IPM regularization weight (0 = no regularization)
        n_epochs: maximum training epochs
        lr: learning rate
        weight_decay: L2 regularization
        early_stopping: stop if validation loss plateaus
        patience: epochs to wait before stopping
        verbose: print training progress
        return_model: if True, also return the fitted GAT model
    
    Returns:
        dict with keys:
            X_GAT_root: shared backbone embeddings (n x rep_dim)
            X_GAT_leaf: list of k embeddings (n x last_hidden_dim each), OR
                       if d1, d2 specified: tuple (X1_leaf, X2_leaf)
            leaf_dim: dimension of leaf embeddings
            model: fitted GATOutcomeModel (only if return_model=True)
    
    Example:
        # Get embeddings for GRR with linear estimator
        emb = GAT_embedding(X, A, D, Y, k=3, d1=0, d2=2)
        
        # Use leaf embeddings for GRR (level-specific representations)
        result = grr_estimator(Y, D, X, d1=0, d2=2, pi_id_all=pi, Kn=Kn,
                               method='linear', 
                               X1=emb['X_GAT_leaf'][0],  # embedding for d1=0
                               X2=emb['X_GAT_leaf'][1])  # embedding for d2=2
    """
    # Import GAT-embedder module
    try:
        from importlib import import_module
        import sys
        import os
        
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import GATOutcomeModel from GAT-embedder
        # Handle the hyphen in filename
        import importlib.util
        gat_path = os.path.join(current_dir, 'GAT-embedder.py')
        spec = importlib.util.spec_from_file_location("gat_embedder", gat_path)
        gat_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gat_module)
        
        GATOutcomeModel = gat_module.GATOutcomeModel
        
    except Exception as e:
        raise ImportError(
            f"Could not import GAT-embedder module: {e}. "
            "Make sure GAT-embedder.py is in the same directory as GRR.py "
            "and torch_geometric is installed."
        )
    
    # Create and fit the GAT model
    model = GATOutcomeModel(
        input_dim=X.shape[1],
        k=k,
        gat_hidden_dims=gat_hidden_dims,
        rep_dim=rep_dim,
        head_hidden_dims=head_hidden_dims,
        heads=heads,
        dropout=dropout,
        lambda_ipm=lambda_ipm
    )
    
    model.fit(
        X, A, D, Y,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay,
        early_stopping=early_stopping,
        patience=patience,
        verbose=verbose
    )
    
    # Extract embeddings
    X_GAT_root, X_GAT_leaf_all = model.get_all_embeddings(X, A)
    leaf_dim = model.get_leaf_embedding_dim()
    
    # If d1, d2 specified, return only those two leaf embeddings
    if d1 is not None and d2 is not None:
        X_GAT_leaf = (X_GAT_leaf_all[d1], X_GAT_leaf_all[d2])
    else:
        X_GAT_leaf = X_GAT_leaf_all
    
    result = {
        'X_GAT_root': X_GAT_root,
        'X_GAT_leaf': X_GAT_leaf,
        'leaf_dim': leaf_dim,
        'rep_dim': rep_dim,
    }
    
    if return_model:
        result['model'] = model
    
    if verbose:
        print(f"\nGAT Embedding Summary:")
        print(f"  X_GAT_root shape: {X_GAT_root.shape} (shared backbone)")
        if d1 is not None and d2 is not None:
            print(f"  X_GAT_leaf shapes: {X_GAT_leaf[0].shape}, {X_GAT_leaf[1].shape} "
                  f"(for d1={d1}, d2={d2})")
        else:
            print(f"  X_GAT_leaf: {len(X_GAT_leaf)} embeddings, each shape {X_GAT_leaf[0].shape}")
    
    return result


# =============================================================================
# PNA Embedding Function
# =============================================================================

def PNA_embedding(X, A, D, Y, k, d1=None, d2=None,
                  pna_hidden_dims=[64, 32],
                  rep_dim=16,
                  head_hidden_dims=[32, 16],
                  aggregators=['mean', 'sum', 'max', 'std'],
                  scalers=['identity', 'amplification', 'attenuation'],
                  dropout=0.1,
                  lambda_ipm=0.0,
                  n_epochs=300,
                  lr=0.001,
                  weight_decay=1e-4,
                  early_stopping=True,
                  patience=50,
                  verbose=True,
                  return_model=False):
    """
    Train a PNA model and extract embeddings for GRR estimation.
    
    PNA (Principal Neighbourhood Aggregation) is superior to GAT when outcomes 
    depend on node degree because:
    - GAT's softmax attention normalizes away degree information
    - PNA uses multiple aggregators (mean, sum, max, std) that preserve degree
    - PNA uses degree scalers (amplification, attenuation) to explicitly inject degree info
    
    This function trains a PNA network using the PNA-embedder module and returns 
    two sets of embeddings:
    
    1. X_PNA_root: Shared backbone embeddings from the PNA encoder (last layer output,
       before the outcome heads). Shape: (n, rep_dim)
       
    2. X_PNA_leaf: Per-exposure-level embeddings from the last hidden layer of 
       each outcome head MLP. Returns a list of k arrays, each of shape 
       (n, last_hidden_dim), or if d1/d2 are specified, returns just 
       (X1_leaf, X2_leaf) for the two specified exposure levels.
    
    Args:
        X: node covariates (n x p), numpy array
        A: adjacency matrix (n x n), numpy array
        D: exposure levels (n,), numpy array with values in {0, ..., k-1}
        Y: observed outcomes (n,), numpy array
        k: number of exposure levels
        d1: first exposure level for comparison (optional)
        d2: second exposure level for comparison (optional)
        pna_hidden_dims: hidden dimensions for PNA encoder
        rep_dim: dimension of PNA output representations
        head_hidden_dims: hidden dimensions for outcome MLP heads
        aggregators: PNA aggregation functions (default: mean, sum, max, std)
        scalers: PNA degree scalers (default: identity, amplification, attenuation)
        dropout: dropout rate
        lambda_ipm: IPM regularization weight (0 = no regularization)
        n_epochs: maximum training epochs
        lr: learning rate
        weight_decay: L2 regularization
        early_stopping: stop if validation loss plateaus
        patience: epochs to wait before stopping
        verbose: print training progress
        return_model: if True, also return the fitted PNA model
    
    Returns:
        dict with keys:
            X_PNA_root: shared backbone embeddings (n x rep_dim)
            X_PNA_leaf: list of k embeddings (n x last_hidden_dim each), OR
                       if d1, d2 specified: tuple (X1_leaf, X2_leaf)
            leaf_dim: dimension of leaf embeddings
            model: fitted PNAOutcomeModel (only if return_model=True)
    
    Example:
        # Get embeddings for GRR with linear estimator
        emb = PNA_embedding(X, A, D, Y, k=3, d1=0, d2=2)
        
        # Use leaf embeddings for GRR (level-specific representations)
        result = grr_estimator(Y, D, X, d1=0, d2=2, pi_id_all=pi, Kn=Kn,
                               method='linear', 
                               X1=emb['X_PNA_leaf'][0],  # embedding for d1=0
                               X2=emb['X_PNA_leaf'][1])  # embedding for d2=2
    """
    # Import PNA-embedder module
    try:
        from importlib import import_module
        import sys
        import os
        
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import PNAOutcomeModel from PNA-embedder
        # Handle the hyphen in filename
        import importlib.util
        pna_path = os.path.join(current_dir, 'PNA-embedder.py')
        spec = importlib.util.spec_from_file_location("pna_embedder", pna_path)
        pna_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pna_module)
        
        PNAOutcomeModel = pna_module.PNAOutcomeModel
        compute_degree_histogram = pna_module.compute_degree_histogram
        
    except Exception as e:
        raise ImportError(
            f"Could not import PNA-embedder module: {e}. "
            "Make sure PNA-embedder.py is in the same directory as GRR.py "
            "and torch_geometric is installed."
        )
    
    # Compute degree histogram (required for PNA)
    deg = compute_degree_histogram(A)
    
    # Create and fit the PNA model
    model = PNAOutcomeModel(
        input_dim=X.shape[1],
        k=k,
        deg=deg,
        pna_hidden_dims=pna_hidden_dims,
        rep_dim=rep_dim,
        head_hidden_dims=head_hidden_dims,
        aggregators=aggregators,
        scalers=scalers,
        dropout=dropout,
        lambda_ipm=lambda_ipm
    )
    
    model.fit(
        X, A, D, Y,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay,
        early_stopping=early_stopping,
        patience=patience,
        verbose=verbose
    )
    
    # Extract embeddings
    X_PNA_root, X_PNA_leaf_all = model.get_all_embeddings(X, A)
    leaf_dim = model.get_leaf_embedding_dim()
    
    # If d1, d2 specified, return only those two leaf embeddings
    if d1 is not None and d2 is not None:
        X_PNA_leaf = (X_PNA_leaf_all[d1], X_PNA_leaf_all[d2])
    else:
        X_PNA_leaf = X_PNA_leaf_all
    
    result = {
        'X_PNA_root': X_PNA_root,
        'X_PNA_leaf': X_PNA_leaf,
        'leaf_dim': leaf_dim,
        'rep_dim': rep_dim,
    }
    
    if return_model:
        result['model'] = model
    
    if verbose:
        print(f"\nPNA Embedding Summary:")
        print(f"  X_PNA_root shape: {X_PNA_root.shape} (shared backbone)")
        print(f"  Aggregators: {aggregators}")
        print(f"  Scalers: {scalers}")
        if d1 is not None and d2 is not None:
            print(f"  X_PNA_leaf shapes: {X_PNA_leaf[0].shape}, {X_PNA_leaf[1].shape} "
                  f"(for d1={d1}, d2={d2})")
        else:
            print(f"  X_PNA_leaf: {len(X_PNA_leaf)} embeddings, each shape {X_PNA_leaf[0].shape}")
    
    return result



