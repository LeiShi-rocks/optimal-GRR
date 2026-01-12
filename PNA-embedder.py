"""
Principal Neighbourhood Aggregation (PNA) for Causal Inference in Network Settings.

This module implements a PNA-based model that:
1. Learns node representations using PNA which PRESERVES DEGREE INFORMATION
2. Predicts potential outcomes for each exposure level using separate MLP heads
3. Uses IPM (Integral Probability Metric) regularization to balance representations

Why PNA over GAT?
    - GAT uses softmax attention which normalizes away degree information
    - PNA combines multiple aggregators (mean, sum, max, std) with degree scalers
    - Degree scalers explicitly inject degree information: amplification (x * log(deg+1))
      and attenuation (x / log(deg+1))
    - This is crucial when outcomes depend on node degree (common in network settings)

Architecture:
    Input: Node covariates X (n x p), Adjacency matrix A (n x n), Exposure D (n,)
    PNA Encoder: Multiple PNA layers to learn representations Phi(X, A)
    MLP Heads: One head per exposure level to predict Y(d) from Phi
    
Loss Function:
    L = MSE(y_hat, y) + lambda_ipm * IPM(Phi | D=d1, Phi | D=d2)
    
where IPM is implemented as Maximum Mean Discrepancy (MMD).

Reference:
    Corso et al. "Principal Neighbourhood Aggregation for Graph Nets" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union

# Check for torch_geometric
try:
    from torch_geometric.nn import PNAConv
    from torch_geometric.utils import dense_to_sparse, degree
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


# =============================================================================
# PNA Encoder
# =============================================================================

class PNAEncoder(nn.Module):
    """
    Principal Neighbourhood Aggregation encoder for learning node representations.
    
    Unlike GAT which uses softmax attention (normalizing away degree info), PNA:
    - Uses multiple aggregators: mean, sum, max, std
    - Uses degree scalers: identity, amplification, attenuation
    - Explicitly preserves and leverages degree information
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 output_dim: int = 16, deg: torch.Tensor = None,
                 aggregators: List[str] = ['mean', 'sum', 'max', 'std'],
                 scalers: List[str] = ['identity', 'amplification', 'attenuation'],
                 dropout: float = 0.1):
        """
        Args:
            input_dim: dimension of input node features
            hidden_dims: list of hidden layer dimensions
            output_dim: dimension of output representations (Phi)
            deg: degree histogram tensor (required for PNA normalization)
            aggregators: aggregation functions to use
            scalers: degree scalers to use
            dropout: dropout rate
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for PNA. "
                "Install with: pip install torch-geometric"
            )
        
        if deg is None:
            raise ValueError(
                "PNA requires degree histogram 'deg'. "
                "Compute with: deg = torch_geometric.utils.degree(edge_index[1], num_nodes)"
            )
        
        self.aggregators = aggregators
        self.scalers = scalers
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Build PNA layers
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.layers.append(
                PNAConv(
                    in_channels=prev_dim,
                    out_channels=h_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    edge_dim=None,  # No edge features
                    towers=1,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False
                )
            )
            
            # LayerNorm after each PNA layer
            self.norms.append(nn.LayerNorm(h_dim))
            prev_dim = h_dim
        
        # Final projection to representation dimension
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PNA encoder.
        
        Args:
            x: node features (n x input_dim)
            edge_index: edge indices (2 x num_edges) in COO format
        
        Returns:
            phi: learned representations (n x output_dim)
        """
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)
        
        phi = self.output_proj(x)
        return phi


# =============================================================================
# Outcome Prediction Heads (same as GAT)
# =============================================================================

class OutcomeHead(nn.Module):
    """
    MLP head for predicting outcomes given representations.
    Each exposure level has its own head to model heterogeneous effects.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [32, 16], 
                 dropout: float = 0.1):
        """
        Args:
            input_dim: dimension of input representations (Phi)
            hidden_dims: hidden layer dimensions for MLP
            dropout: dropout rate
        """
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.last_hidden_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # Build hidden layers (all but the final output layer)
        hidden_layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            hidden_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        self.hidden_net = nn.Sequential(*hidden_layers) if hidden_layers else nn.Identity()
        
        # Output layer (single outcome value)
        self.output_layer = nn.Linear(prev_dim, 1)
    
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Predict outcome from representation.
        
        Args:
            phi: node representations (n x input_dim)
        
        Returns:
            y_hat: predicted outcomes (n,)
        """
        h = self.hidden_net(phi)
        return self.output_layer(h).squeeze(-1)
    
    def get_last_hidden(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Get the last hidden layer embedding (before the output layer).
        
        Args:
            phi: node representations (n x input_dim)
        
        Returns:
            h: last hidden layer output (n x last_hidden_dim)
        """
        return self.hidden_net(phi)


# =============================================================================
# IPM Regularizer (Maximum Mean Discrepancy) - same as GAT
# =============================================================================

class IPMRegularizer(nn.Module):
    """
    Integral Probability Metric (IPM) regularizer using Maximum Mean Discrepancy.
    
    Encourages similar representation distributions across exposure groups,
    which is crucial for valid causal inference.
    
    MMD^2(P, Q) = E_{x,x'~P}[k(x,x')] - 2*E_{x~P,y~Q}[k(x,y)] + E_{y,y'~Q}[k(y,y')]
    """
    
    def __init__(self, kernel: str = 'rbf', sigma: float = 1.0, 
                 multi_scale: bool = True):
        """
        Args:
            kernel: 'rbf' (Gaussian) or 'linear'
            sigma: base bandwidth for RBF kernel
            multi_scale: if True, use multiple kernel bandwidths
        """
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        self.multi_scale = multi_scale
        
        # Multiple bandwidths for robust MMD estimation
        if multi_scale:
            self.sigmas = [sigma * s for s in [0.1, 0.5, 1.0, 2.0, 5.0]]
        else:
            self.sigmas = [sigma]
    
    def rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor, 
                   sigma: float) -> torch.Tensor:
        """Compute RBF (Gaussian) kernel matrix."""
        # Compute squared Euclidean distances
        X_sqnorms = (X ** 2).sum(dim=1)
        Y_sqnorms = (Y ** 2).sum(dim=1)
        XY = X @ Y.T
        
        # ||x - y||^2 = ||x||^2 - 2<x,y> + ||y||^2
        dists_sq = X_sqnorms.unsqueeze(1) - 2 * XY + Y_sqnorms.unsqueeze(0)
        dists_sq = torch.clamp(dists_sq, min=0)  # Numerical stability
        
        return torch.exp(-dists_sq / (2 * sigma ** 2))
    
    def linear_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute linear kernel matrix."""
        return X @ Y.T
    
    def compute_mmd_single(self, phi_d1: torch.Tensor, phi_d2: torch.Tensor,
                          sigma: float) -> torch.Tensor:
        """Compute MMD^2 for a single kernel bandwidth."""
        n1 = phi_d1.size(0)
        n2 = phi_d2.size(0)
        
        if n1 < 2 or n2 < 2:
            return torch.tensor(0.0, device=phi_d1.device)
        
        if self.kernel == 'rbf':
            K_11 = self.rbf_kernel(phi_d1, phi_d1, sigma)
            K_22 = self.rbf_kernel(phi_d2, phi_d2, sigma)
            K_12 = self.rbf_kernel(phi_d1, phi_d2, sigma)
        else:
            K_11 = self.linear_kernel(phi_d1, phi_d1)
            K_22 = self.linear_kernel(phi_d2, phi_d2)
            K_12 = self.linear_kernel(phi_d1, phi_d2)
        
        # Unbiased MMD^2 estimate (remove diagonal)
        mmd = (K_11.sum() - K_11.trace()) / (n1 * (n1 - 1) + 1e-8)
        mmd += (K_22.sum() - K_22.trace()) / (n2 * (n2 - 1) + 1e-8)
        mmd -= 2 * K_12.mean()
        
        return mmd
    
    def compute_mmd(self, phi_d1: torch.Tensor, phi_d2: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between two groups, optionally using multiple scales.
        """
        if self.multi_scale:
            mmd = sum(self.compute_mmd_single(phi_d1, phi_d2, s) 
                     for s in self.sigmas) / len(self.sigmas)
        else:
            mmd = self.compute_mmd_single(phi_d1, phi_d2, self.sigma)
        return mmd
    
    def forward(self, phi: torch.Tensor, D: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute total IPM loss across all pairs of exposure levels.
        
        Args:
            phi: node representations (n x d)
            D: exposure levels (n,), values in {0, 1, ..., k-1}
            k: number of exposure levels
        
        Returns:
            ipm_loss: average MMD across all exposure pairs
        """
        total_ipm = torch.tensor(0.0, device=phi.device)
        n_pairs = 0
        
        # Compute MMD for all pairs of exposure levels
        for d1 in range(k):
            for d2 in range(d1 + 1, k):
                mask_d1 = (D == d1)
                mask_d2 = (D == d2)
                
                # Need at least 2 samples per group for unbiased MMD
                if mask_d1.sum() >= 2 and mask_d2.sum() >= 2:
                    phi_d1 = phi[mask_d1]
                    phi_d2 = phi[mask_d2]
                    total_ipm = total_ipm + self.compute_mmd(phi_d1, phi_d2)
                    n_pairs += 1
        
        if n_pairs > 0:
            return total_ipm / n_pairs
        return total_ipm


# =============================================================================
# Main Model: PNA Outcome Model
# =============================================================================

class PNAOutcomeModel(nn.Module):
    """
    Principal Neighbourhood Aggregation model for outcome prediction in network causal inference.
    
    Key advantage over GAT:
    - PNA preserves degree information through multiple aggregators and degree scalers
    - This is crucial when outcomes depend on node degree (Y = beta_h * H + ...)
    
    This model:
    1. Uses PNA to learn node representations that capture degree information
    2. Has separate MLP heads for each exposure level
    3. Uses IPM (MMD) regularization to balance representations across exposure groups
    
    The objective is:
        L = MSE(y_hat, y) + lambda_ipm * IPM(Phi)
    """
    
    def __init__(self, input_dim: int, k: int,
                 deg: torch.Tensor,
                 pna_hidden_dims: List[int] = [64, 32],
                 rep_dim: int = 16,
                 head_hidden_dims: List[int] = [32, 16],
                 aggregators: List[str] = ['mean', 'sum', 'max', 'std'],
                 scalers: List[str] = ['identity', 'amplification', 'attenuation'],
                 dropout: float = 0.1,
                 ipm_kernel: str = 'rbf',
                 ipm_sigma: float = 1.0,
                 lambda_ipm: float = 1.0,
                 device: Optional[str] = None):
        """
        Args:
            input_dim: dimension of node covariates X
            k: number of exposure levels (D in {0, 1, ..., k-1})
            deg: degree histogram tensor (required for PNA)
            pna_hidden_dims: hidden dimensions for PNA encoder
            rep_dim: dimension of learned representations Phi
            head_hidden_dims: hidden dimensions for outcome MLP heads
            aggregators: PNA aggregation functions
            scalers: PNA degree scalers
            dropout: dropout rate
            ipm_kernel: 'rbf' or 'linear' for IPM
            ipm_sigma: bandwidth for RBF kernel
            lambda_ipm: weight for IPM regularization
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        super().__init__()
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.k = k
        self.lambda_ipm = lambda_ipm
        self.rep_dim = rep_dim
        
        # Move degree histogram to device
        deg = deg.to(self.device)
        
        # PNA encoder for learning representations
        self.encoder = PNAEncoder(
            input_dim=input_dim,
            hidden_dims=pna_hidden_dims,
            output_dim=rep_dim,
            deg=deg,
            aggregators=aggregators,
            scalers=scalers,
            dropout=dropout
        )
        
        # Outcome heads: one MLP per exposure level
        self.outcome_heads = nn.ModuleList([
            OutcomeHead(rep_dim, head_hidden_dims, dropout)
            for _ in range(k)  # Levels 0, 1, ..., k-1
        ])
        
        # IPM regularizer
        self.ipm = IPMRegularizer(kernel=ipm_kernel, sigma=ipm_sigma)
        
        # Move to device
        self.to(self.device)
        
        self._fitted = False
        self._edge_index = None
        
        # Print device info
        print(f"[PNAOutcomeModel] Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"[PNAOutcomeModel] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[PNAOutcomeModel] Exposure levels: 0 to {k-1} ({k} levels)")
        print(f"[PNAOutcomeModel] Aggregators: {aggregators}")
        print(f"[PNAOutcomeModel] Scalers: {scalers}")
    
    def get_representations(self, X: torch.Tensor, 
                           edge_index: torch.Tensor) -> torch.Tensor:
        """Get learned node representations Phi(X, A)."""
        return self.encoder(X, edge_index)
    
    def predict_outcome(self, phi: torch.Tensor, d: int) -> torch.Tensor:
        """Predict outcome Y(d) for a specific exposure level."""
        return self.outcome_heads[d](phi)
    
    def predict_all_outcomes(self, phi: torch.Tensor) -> List[torch.Tensor]:
        """Predict outcomes Y(0), Y(1), ..., Y(k-1) for all exposure levels."""
        return [head(phi) for head in self.outcome_heads]
    
    def forward(self, X: torch.Tensor, edge_index: torch.Tensor
               ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass: compute representations and all potential outcomes.
        
        Args:
            X: node features (n x input_dim)
            edge_index: edge indices (2 x num_edges)
        
        Returns:
            phi: learned representations (n x rep_dim)
            y_hats: list of predicted outcomes for each exposure level
        """
        phi = self.encoder(X, edge_index)
        y_hats = self.predict_all_outcomes(phi)
        return phi, y_hats
    
    def compute_loss(self, X: torch.Tensor, edge_index: torch.Tensor,
                     D: torch.Tensor, Y: torch.Tensor,
                     mask: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss = MSE + lambda * IPM.
        
        Args:
            X: node features (n x input_dim)
            edge_index: edge indices (2 x num_edges)
            D: observed exposure levels (n,)
            Y: observed outcomes (n,)
            mask: optional boolean mask for valid observations (n,)
        
        Returns:
            total_loss: combined loss for backprop
            mse_loss: prediction MSE (for monitoring)
            ipm_loss: IPM regularization (for monitoring)
        """
        # Get representations
        phi = self.encoder(X, edge_index)
        
        # MSE loss: predict Y using the head corresponding to observed D
        mse_loss = torch.tensor(0.0, device=self.device)
        n_obs = 0
        
        for d in range(self.k):
            mask_d = (D == d)
            if mask is not None:
                mask_d = mask_d & mask
            
            n_d = mask_d.sum()
            if n_d > 0:
                y_pred = self.outcome_heads[d](phi[mask_d])
                y_true = Y[mask_d]
                mse_loss = mse_loss + F.mse_loss(y_pred, y_true, reduction='sum')
                n_obs += n_d
        
        if n_obs > 0:
            mse_loss = mse_loss / n_obs
        
        # IPM loss: balance representations across exposure groups
        ipm_loss = self.ipm(phi, D, self.k)
        
        # Total loss
        total_loss = mse_loss + self.lambda_ipm * ipm_loss
        
        return total_loss, mse_loss, ipm_loss
    
    def fit(self, X: np.ndarray, A: np.ndarray, D: np.ndarray, Y: np.ndarray,
            lr: float = 0.001,
            n_epochs: int = 500,
            weight_decay: float = 1e-4,
            early_stopping: bool = True,
            patience: int = 50,
            min_delta: float = 1e-4,
            val_split: float = 0.1,
            verbose: bool = True) -> 'PNAOutcomeModel':
        """
        Fit the model using gradient descent.
        
        Args:
            X: node covariates (n x p), numpy array
            A: adjacency matrix (n x n), numpy array
            D: exposure levels (n,), numpy array with values in {0, ..., k-1}
            Y: observed outcomes (n,), numpy array
            lr: learning rate
            n_epochs: maximum training epochs
            weight_decay: L2 regularization
            early_stopping: stop if validation loss plateaus
            patience: epochs to wait before stopping
            min_delta: minimum improvement for early stopping
            val_split: fraction of data for validation
            verbose: print training progress
        
        Returns:
            self
        """
        # Convert to tensors and move to device
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        D_t = torch.tensor(D, dtype=torch.long, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        
        # Convert adjacency matrix to edge_index (COO format)
        A_tensor = torch.tensor(A, dtype=torch.float32)
        edge_index, _ = dense_to_sparse(A_tensor)
        edge_index = edge_index.to(self.device)
        self._edge_index = edge_index
        
        # Train/validation split
        n = X_t.size(0)
        n_val = max(1, int(n * val_split))
        perm = torch.randperm(n, device=self.device)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        
        train_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        train_mask[train_idx] = True
        val_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        val_mask[val_idx] = True
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, 
                                      weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        if verbose:
            print(f"\nTraining PNAOutcomeModel...")
            print(f"  Train samples: {train_mask.sum().item()}, "
                  f"Val samples: {val_mask.sum().item()}")
        
        for epoch in range(n_epochs):
            # Training step
            self.train()
            optimizer.zero_grad()
            
            train_loss, train_mse, train_ipm = self.compute_loss(
                X_t, edge_index, D_t, Y_t, mask=train_mask
            )
            train_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Validation step
            self.eval()
            with torch.no_grad():
                val_loss, val_mse, val_ipm = self.compute_loss(
                    X_t, edge_index, D_t, Y_t, mask=val_mask
                )
            
            scheduler.step(val_loss)
            
            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            
            # Logging
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1:4d}/{n_epochs}: "
                      f"train_loss={train_loss.item():.4f} "
                      f"(mse={train_mse.item():.4f}, ipm={train_ipm.item():.4f}) | "
                      f"val_loss={val_loss.item():.4f}")
        
        # Restore best model
        if best_state is not None:
            self.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        self._fitted = True
        
        if verbose:
            print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None,
                D: Optional[Union[int, np.ndarray]] = None) -> np.ndarray:
        """
        Predict outcomes.
        
        Args:
            X: node covariates (n x p)
            A: adjacency matrix (optional, uses fitted graph if None)
            D: exposure level(s) to predict for:
               - int: predict Y(d) for all nodes at exposure level d
               - array: predict Y(D_i) for each node's specified exposure
               - None: return all potential outcomes Y(0), Y(1), ..., Y(k-1)
        
        Returns:
            if D is int: predicted outcomes (n,)
            if D is array: predicted outcomes (n,) using each node's exposure
            if D is None: potential outcomes matrix (n x k)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.eval()
        with torch.no_grad():
            # Convert inputs
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            
            # Get edge_index
            if A is not None:
                A_tensor = torch.tensor(A, dtype=torch.float32)
                edge_index, _ = dense_to_sparse(A_tensor)
                edge_index = edge_index.to(self.device)
            else:
                edge_index = self._edge_index
            
            # Get representations
            phi = self.encoder(X_t, edge_index)
            
            if D is None:
                # Return all potential outcomes
                Y_pot = torch.zeros(X_t.size(0), self.k, device=self.device)
                for d in range(self.k):
                    Y_pot[:, d] = self.outcome_heads[d](phi)
                return Y_pot.cpu().numpy()
            
            elif isinstance(D, int):
                # Predict for specific exposure level
                y_hat = self.outcome_heads[D](phi)
                return y_hat.cpu().numpy()
            
            else:
                # Predict using each node's specified exposure
                D_t = torch.tensor(D, dtype=torch.long, device=self.device)
                y_hat = torch.zeros(X_t.size(0), device=self.device)
                
                for d in range(self.k):
                    mask_d = (D_t == d)
                    if mask_d.sum() > 0:
                        y_hat[mask_d] = self.outcome_heads[d](phi[mask_d])
                
                return y_hat.cpu().numpy()
    
    def get_node_representations(self, X: np.ndarray, 
                                  A: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get learned node representations.
        
        Args:
            X: node covariates (n x p)
            A: adjacency matrix (optional)
        
        Returns:
            phi: learned representations (n x rep_dim)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            
            if A is not None:
                A_tensor = torch.tensor(A, dtype=torch.float32)
                edge_index, _ = dense_to_sparse(A_tensor)
                edge_index = edge_index.to(self.device)
            else:
                edge_index = self._edge_index
            
            phi = self.encoder(X_t, edge_index)
            return phi.cpu().numpy()
    
    def get_all_embeddings(self, X: np.ndarray, 
                           A: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Get both root embeddings (shared backbone) and leaf embeddings (per-head).
        
        Args:
            X: node covariates (n x p)
            A: adjacency matrix (optional)
        
        Returns:
            X_PNA_root: shared backbone embeddings from PNA encoder (n x rep_dim)
            X_PNA_leaf: list of k embeddings, one per exposure level (n x last_hidden_dim)
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            
            if A is not None:
                A_tensor = torch.tensor(A, dtype=torch.float32)
                edge_index, _ = dense_to_sparse(A_tensor)
                edge_index = edge_index.to(self.device)
            else:
                edge_index = self._edge_index
            
            # Get root embeddings (shared backbone from PNA encoder)
            phi = self.encoder(X_t, edge_index)
            X_PNA_root = phi.cpu().numpy()
            
            # Get leaf embeddings (last hidden layer from each outcome head)
            X_PNA_leaf = []
            for d in range(self.k):
                leaf_emb = self.outcome_heads[d].get_last_hidden(phi)
                X_PNA_leaf.append(leaf_emb.cpu().numpy())
            
            return X_PNA_root, X_PNA_leaf
    
    def get_leaf_embedding_dim(self) -> int:
        """Get the dimension of leaf embeddings."""
        return self.outcome_heads[0].last_hidden_dim


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_degree_histogram(A: np.ndarray, max_degree: int = None) -> torch.Tensor:
    """
    Compute degree histogram for PNA initialization.
    
    Args:
        A: adjacency matrix (n x n)
        max_degree: cap on maximum degree (optional)
    
    Returns:
        deg: degree histogram tensor
    """
    A_tensor = torch.tensor(A, dtype=torch.float32)
    edge_index, _ = dense_to_sparse(A_tensor)
    
    # Compute degree for each node
    deg = degree(edge_index[1], num_nodes=A.shape[0], dtype=torch.long)
    
    # Cap at max_degree if specified
    if max_degree is not None:
        deg = torch.clamp(deg, max=max_degree)
    
    # Create histogram
    max_deg = deg.max().item() + 1
    deg_hist = torch.zeros(max_deg, dtype=torch.long)
    for d in deg:
        deg_hist[d.item()] += 1
    
    return deg_hist


def fit_pna_outcome_model(X: np.ndarray, A: np.ndarray, D: np.ndarray, 
                          Y: np.ndarray, k: int, **kwargs) -> PNAOutcomeModel:
    """
    Convenience function to fit a PNA outcome model.
    
    Args:
        X: node covariates (n x p)
        A: adjacency matrix (n x n)
        D: exposure levels (n,), values in {0, 1, ..., k-1}
        Y: observed outcomes (n,)
        k: number of exposure levels
        **kwargs: passed to PNAOutcomeModel and fit()
    
    Returns:
        Fitted PNAOutcomeModel
    """
    # Compute degree histogram
    deg = compute_degree_histogram(A)
    
    # Split kwargs between model init and fit
    model_kwargs = {
        'pna_hidden_dims': kwargs.pop('pna_hidden_dims', [64, 32]),
        'rep_dim': kwargs.pop('rep_dim', 16),
        'head_hidden_dims': kwargs.pop('head_hidden_dims', [32, 16]),
        'aggregators': kwargs.pop('aggregators', ['mean', 'sum', 'max', 'std']),
        'scalers': kwargs.pop('scalers', ['identity', 'amplification', 'attenuation']),
        'dropout': kwargs.pop('dropout', 0.1),
        'ipm_kernel': kwargs.pop('ipm_kernel', 'rbf'),
        'ipm_sigma': kwargs.pop('ipm_sigma', 1.0),
        'lambda_ipm': kwargs.pop('lambda_ipm', 1.0),
        'device': kwargs.pop('device', None),
    }
    
    model = PNAOutcomeModel(input_dim=X.shape[1], k=k, deg=deg, **model_kwargs)
    model.fit(X, A, D, Y, **kwargs)
    
    return model


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example using out1 setup from simulation-GRR.ipynb
    print("PNA-Embedder Example (out1 configuration)")
    print("=" * 60)
    print("\nPNA vs GAT: PNA preserves degree information!")
    print("  - GAT: softmax attention normalizes away degree")
    print("  - PNA: uses sum aggregator + degree scalers")
    print("=" * 60)
    
    # Import DGP functions
    from dgp import (
        generate_graph,
        generate_covariates,
        generate_potential_outcomes,
        assign_treatment,
        compute_exposure,
        observed_outcome
    )
    
    # out1 parameters from simulation-GRR.ipynb
    n = 2000
    avg_degree = 3.0
    max_degree = 9
    k = 3  # Number of exposure levels (0, 1, 2)
    treated_fraction = 1/3
    d1, d2 = 0, 2  # Exposure levels to compare
    seed_graph = 123
    seed_cov = 1234
    seed_out = 999
    
    print(f"\nGenerating network with n={n}, avg_degree={avg_degree}, k={k}")
    
    # Generate graph and covariates using DGP
    G, H = generate_graph(n=n, avg_degree=avg_degree, max_degree=max_degree, seed=seed_graph)
    X, X_noi = generate_covariates(H, seed=seed_cov)
    X_run = X[:, [0, 3]]  # [intercept, X3] - note: missing degree info!
    Y_pot_true = generate_potential_outcomes(H, X_noi, k=k, seed=seed_out)
    
    # Compute true treatment effect
    true_tau = float(np.mean(Y_pot_true[:, d1] - Y_pot_true[:, d2]))
    print(f"True tau (d={d1} vs d={d2}): {true_tau:.4f}")
    
    # Generate treatment assignment and exposure
    A_treat = assign_treatment(n, treated_fraction=treated_fraction, seed=42)
    D = compute_exposure(G, A_treat, k)
    Y = observed_outcome(Y_pot_true, D, k=k)
    
    # Print exposure distribution
    print(f"\nExposure distribution:")
    for d in range(k):
        count = (D == d).sum()
        print(f"  D={d}: {count} ({100*count/n:.1f}%)")
    
    # Print degree statistics
    print(f"\nDegree statistics:")
    print(f"  Mean degree: {H.mean():.2f}")
    print(f"  Std degree: {H.std():.2f}")
    print(f"  Min/Max degree: {H.min()}/{H.max()}")
    
    # Use adjacency matrix G for PNA
    A = G.astype(float)
    
    print(f"\nTraining PNAOutcomeModel...")
    print(f"  Input dim: {X_run.shape[1]}")
    print(f"  Exposure levels: 0 to {k-1} ({k} levels)")
    
    # Fit model
    model = fit_pna_outcome_model(
        X_run, A, D, Y, k=k,
        pna_hidden_dims=[64, 32],
        rep_dim=16,
        head_hidden_dims=[32, 16],
        dropout=0.1,
        lambda_ipm=0.0,  # Try without IPM first
        n_epochs=300,
        lr=0.001,
        weight_decay=1e-4,
        early_stopping=True,
        patience=50,
        verbose=True
    )
    
    # Predict potential outcomes
    Y_pot_pred = model.predict(X_run)
    print(f"\nPredicted potential outcomes shape: {Y_pot_pred.shape}")
    for d in range(k):
        print(f"  Mean Y({d}): predicted={Y_pot_pred[:, d].mean():.3f}, "
              f"true={Y_pot_true[:, d].mean():.3f}")
    
    # Estimate treatment effect (tau = E[Y(d1)] - E[Y(d2)])
    tau_hat = Y_pot_pred[:, d1].mean() - Y_pot_pred[:, d2].mean()
    print(f"\nTreatment effect estimation:")
    print(f"  tau_hat (d={d1} vs d={d2}): {tau_hat:.4f}")
    print(f"  true_tau:                   {true_tau:.4f}")
    print(f"  bias:                       {tau_hat - true_tau:.4f}")
    
    # Compute R² for PNA predictions
    Y_pred_pna = np.zeros(n)
    for d in range(k):
        mask_d = (D == d)
        if mask_d.sum() > 0:
            Y_pred_pna[mask_d] = model.predict(X_run, D=d)[mask_d]
    
    ss_res = np.sum((Y - Y_pred_pna) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2_pna = 1 - ss_res / ss_tot
    print(f"\nR² for outcome prediction:")
    print(f"  PNA model R²: {r2_pna:.4f}")
    
    # Compare with linear model
    from sklearn.linear_model import LinearRegression
    Y_pred_linear = np.zeros(n)
    for d in range(k):
        mask_d = (D == d)
        if mask_d.sum() > 5:
            lr = LinearRegression().fit(X_run[mask_d], Y[mask_d])
            Y_pred_linear[mask_d] = lr.predict(X_run[mask_d])
        else:
            Y_pred_linear[mask_d] = Y[mask_d].mean()
    
    ss_res_lin = np.sum((Y - Y_pred_linear) ** 2)
    r2_linear = 1 - ss_res_lin / ss_tot
    print(f"  Linear model R²: {r2_linear:.4f}")
    
    # Get representations
    phi = model.get_node_representations(X_run)
    print(f"\nRepresentations shape: {phi.shape}")
    
    # Show that PNA captures degree
    print(f"\nDegree capture test:")
    print(f"  Correlation(phi[0], degree): {np.corrcoef(phi[:, 0], H)[0, 1]:.4f}")

