"""
Functions for IPW, Hajek, and AIPW estimation with variance computation.
Extracted from effect.py to avoid path dependencies.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def load_second_order_probs(combination, matrix_dir):
    """Load sparse second-order inclusion probability matrix for a given combination."""
    a, s, h = combination
    filename = f"{matrix_dir}/P_A{a}_S{s}_H{h}_sparse.csv"
    sparse_df = pd.read_csv(filename)
    # Create dictionary: (idx_i, idx_j) -> probability
    prob_dict = {}
    for _, row in sparse_df.iterrows():
        idx_i = int(row['village_i_idx'])
        idx_j = int(row['village_j_idx'])
        prob = row['probability']
        prob_dict[(idx_i, idx_j)] = prob
    return prob_dict


def general_weighted_ipw_hajek(Y, A, S, H, propensity_scores, target_combination, weights):
    """
    General weighted IPW/Hajek estimator
    
    Parameters:
    -----------
    Y : array-like
        Outcome variable for each unit (e.g., village-level profits)
    A : array-like
        Treatment indicator (0 or 1) for each unit
    S : array-like
        Sublocation spillover indicator (0 or 1) for each unit
        S = 1 if more than half of other villages in the same sublocation are treated
    H : array-like
        Neighborhood spillover indicator (0 or 1) for each unit
        H = 1 if more than half of the nearest villages OUTSIDE own sublocation are treated
    propensity_scores : DataFrame
        Propensity score matrix with columns for each (A,S,H) combination
        Columns should be named like '(A=0,S=0,H=0)', '(A=1,S=1,H=1)', etc.
    target_combination : tuple
        Target combination (a, s, h) to estimate, e.g., (1, 0, 1)
    weights : array-like
        Weight gamma_i for each unit
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'ipw': IPW estimate
        - 'hajek': Hajek estimate
        - 'ipw_numerator_sum': Sum of IPW numerator terms
        - 'hajek_denominator': Denominator of Hajek estimator
        - 'n_units_in_target': Number of units with (A,S,H) = target
        - 'effective_n': Effective sample size after weighting
    """
    # Convert inputs to numpy arrays
    Y = np.array(Y)
    A = np.array(A)
    S = np.array(S)
    H = np.array(H)
    weights = np.array(weights)
    
    # Extract target combination
    a, s, h = target_combination
    
    # Create column name for the target combination
    col_name = f'(A={a},S={s},H={h})'
    
    # Check if column exists in propensity_scores
    if col_name not in propensity_scores.columns:
        raise ValueError(f"Column '{col_name}' not found in propensity_scores. Available columns: {propensity_scores.columns.tolist()}")
    
    # Get propensity scores for the target combination
    ps = propensity_scores[col_name].values
    
    # Avoid division by zero - add small epsilon to propensity scores if needed
    ps = np.where(ps > 0, ps, 1e-10)
    
    # Create indicator for units with (A, S, H) = (a, s, h)
    indicator = ((A == a) & (S == s) & (H == h)).astype(float)
    
    # IPW estimator: mean(Y * indicator * gamma_i / propensity_score)
    ipw_terms = Y * indicator * weights / ps
    ipw_estimate = np.mean(ipw_terms)
    
    # Hajek estimator
    # Numerator: same as IPW (but we'll compute it as sum/n for clarity)
    hajek_numerator = np.mean(ipw_terms)
    
    # Denominator: mean(indicator * gamma_i / propensity_score) / mean(gamma_i)
    denominator_terms = indicator * weights / ps
    hajek_denominator_numerator = np.mean(denominator_terms)
    hajek_denominator = hajek_denominator_numerator / np.mean(weights)
    
    # Hajek estimate
    if hajek_denominator == 0:
        hajek_estimate = np.nan
        print(f"Warning: Hajek denominator is 0 for target combination {target_combination}")
    else:
        hajek_estimate = hajek_numerator / hajek_denominator
    
    # Calculate effective sample size
    effective_n = (np.sum(denominator_terms))**2 / np.sum(denominator_terms**2)
    
    return {
        'ipw': ipw_estimate,
        'hajek': hajek_estimate,
        'ipw_numerator_sum': np.sum(ipw_terms),
        'hajek_denominator': hajek_denominator,
        'n_units_in_target': int(np.sum(indicator)),
        'effective_n': effective_n
    }


def compute_variance_ipw_hajek(Y, A, S, H, propensity_scores, target_combination, weights, village_idx_map, matrix_dir):
    """
    Compute variance estimates for IPW and Hajek estimators using second-order inclusion probabilities.
    
    Parameters:
    -----------
    Y : array-like
        Outcome variable
    A, S, H : array-like
        Treatment and spillover indicators
    propensity_scores : DataFrame
        First-order propensity scores
    target_combination : tuple
        Target (a, s, h) combination
    weights : array-like
        Weights gamma_i
    village_idx_map : dict
        Mapping from data index to village index (for second-order probs)
    matrix_dir : str
        Directory containing second-order probability matrices
    
    Returns:
    --------
    dict : Dictionary with 'ipw_var', 'hajek_var', 'ipw_se', 'hajek_se'
    """
    # Convert to arrays
    Y = np.array(Y)
    A = np.array(A)
    S = np.array(S)
    H = np.array(H)
    weights = np.array(weights)
    
    n = len(Y)
    a, s, h = target_combination
    
    # Get first-order propensity scores
    col_name = f'(A={a},S={s},H={h})'
    p_i = propensity_scores[col_name].values
    p_i = np.where(p_i > 0, p_i, 1e-10)  # Avoid division by zero
    
    # Load second-order probabilities (sparse)
    p_ij_dict = load_second_order_probs(target_combination, matrix_dir)
    
    # Create indicator
    indicator = ((A == a) & (S == s) & (H == h)).astype(float)
    
    # Compute IPW reweighted outcomes: Yhat_i = Y_i * indicator_i / p_i
    Yhat = Y * indicator / p_i
    
    # IPW Variance
    # Diagonal terms: w_ii = (1 - p_ii), sum w_ii * Yhat_i^2
    diagonal_sum = 0.0
    for i in range(n):
        w_ii = 1 - p_i[i]
        diagonal_sum += w_ii * (Yhat[i] ** 2) * (weights[i] ** 2)
    
    # Off-diagonal terms (only for dependent pairs)
    offdiag_sum = 0.0
    
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle (i < j)
            # Get village indices
            vi = village_idx_map.get(i)
            vj = village_idx_map.get(j)
            
            if vi is None or vj is None:
                continue
            
            # Check if pair exists in sparse matrix
            pair_key = (min(vi, vj), max(vi, vj))
            
            if pair_key in p_ij_dict:
                p_ij = p_ij_dict[pair_key]
                
                if p_ij >= 1e-4:
                    # Use covariance formula
                    w_ij = (p_ij - p_i[i] * p_i[j]) / p_ij
                    offdiag_sum += 2 * w_ij * Yhat[i] * Yhat[j] * weights[i] * weights[j]  # 2x for symmetry
                else:
                    # Conservative estimate for very small p_ij
                    offdiag_sum += (p_i[i]/2 * Yhat[i]**2 * (weights[i]**2) + p_i[j]/2 * Yhat[j]**2 * (weights[j]**2))
    
    ipw_variance = (diagonal_sum + offdiag_sum) / (n ** 2)
    
    # Hajek Variance
    # First compute Hajek estimate
    hajek_est = general_weighted_ipw_hajek(Y, A, S, H, propensity_scores, target_combination, weights)['hajek']
    mean_weights = np.mean(weights)
    
    # Demean outcomes: Yhat_i_demean = (Y_i - hajek_est/mean_weights) * indicator_i / p_i
    Y_demeaned = Y - hajek_est / mean_weights
    Yhat_demean = Y_demeaned * indicator / p_i
    
    # Diagonal terms
    diagonal_sum_hajek = 0.0
    for i in range(n):
        w_ii = 1 - p_i[i]
        diagonal_sum_hajek += w_ii * (Yhat_demean[i] ** 2) * (weights[i] ** 2)
    
    # Off-diagonal terms
    offdiag_sum_hajek = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            vi = village_idx_map.get(i)
            vj = village_idx_map.get(j)
            
            if vi is None or vj is None:
                continue
            
            pair_key = (min(vi, vj), max(vi, vj))
            
            if pair_key in p_ij_dict:
                p_ij = p_ij_dict[pair_key]
                
                if p_ij >= 1e-4:
                    w_ij = (p_ij - p_i[i] * p_i[j]) / p_ij
                    offdiag_sum_hajek += 2 * w_ij * Yhat_demean[i] * Yhat_demean[j] * weights[i] * weights[j]
                else:
                    # offdiag_sum_hajek += (p_i[i]/2 * Yhat_demean[i]**2 * (weights[i]**2) + p_i[j]/2 * Yhat_demean[j]**2 * (weights[j]**2))
                    offdiag_sum_hajek += 0
    
    hajek_variance = (diagonal_sum_hajek + offdiag_sum_hajek) / (n ** 2) 
    
    return {
        'ipw_var': ipw_variance,
        'hajek_var': hajek_variance,
        'ipw_se': np.sqrt(ipw_variance) if ipw_variance > 0 else 0,
        'hajek_se': np.sqrt(hajek_variance) if hajek_variance > 0 else 0
    }


def fit_outcome_model(Y, A, S, H, X_covariates):
    """
    Fit outcome model with centered covariates and full interaction with (A,S,H).
    
    Model: Y_i ~ sum_(a,s,h) I(A_i=a,S_i=s,H_i=h) + X_i_center + X_i_center * sum_(a,s,h) I(A_i=a,S_i=s,H_i=h)
    
    Parameters:
    -----------
    Y : array-like
        Outcome variable
    A, S, H : array-like
        Treatment and spillover indicators
    X_covariates : DataFrame or array
        Covariates (will be centered)
    
    Returns:
    --------
    model : fitted LinearRegression model
    X_centered : array
        Centered covariates
    design_matrix : array
        Full design matrix used for fitting
    """
    Y = np.array(Y)
    A = np.array(A)
    S = np.array(S)
    H = np.array(H)
    
    # Standardize covariates (center and scale)
    if isinstance(X_covariates, pd.DataFrame):
        X = X_covariates.values
    else:
        X = np.array(X_covariates)
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    # Avoid division by zero for constant columns
    X_std = np.where(X_std > 1e-10, X_std, 1)
    X_centered = (X - X_mean) / X_std
    
    n = len(Y)
    n_covariates = X_centered.shape[1]
    
    # Create indicators for all 8 (a,s,h) combinations
    indicators = np.zeros((n, 8))
    idx_map = {}
    for i_comb, (a, s, h) in enumerate([(a, s, h) for a in [0,1] for s in [0,1] for h in [0,1]]):
        indicators[:, i_comb] = ((A == a) & (S == s) & (H == h)).astype(float)
        idx_map[(a, s, h)] = i_comb
    
    # Build design matrix: [indicators, X_centered, X_centered * indicators]
    # Note: We exclude one indicator to avoid perfect collinearity (use (0,0,0) as baseline)
    indicators_for_model = indicators[:, 1:]  # Exclude (0,0,0) indicator
    
    # Interactions: X_centered * each indicator (excluding baseline)
    interactions = []
    for i_comb in range(1, 8):
        for j_cov in range(n_covariates):
            interactions.append(X_centered[:, j_cov] * indicators[:, i_comb])
    interactions = np.column_stack(interactions) if interactions else np.zeros((n, 0))
    
    # Full design matrix
    design_matrix = np.column_stack([indicators_for_model, X_centered, interactions])
    
    # Fit model
    model = LinearRegression(fit_intercept=True)
    model.fit(design_matrix, Y)
    
    return model, X_centered, design_matrix, idx_map


def predict_outcome_model(model, X_centered, target_combination, idx_map):
    """
    Get predictions from outcome model for a specific (a,s,h) combination.
    
    Parameters:
    -----------
    model : fitted LinearRegression
    X_centered : array
        Centered covariates
    target_combination : tuple
        (a, s, h) combination to predict for
    idx_map : dict
        Mapping from (a,s,h) to indicator index
    
    Returns:
    --------
    predictions : array
        Predicted outcomes Y_pred(a,s,h) for all units
    """
    a, s, h = target_combination
    n = X_centered.shape[0]
    n_covariates = X_centered.shape[1]
    
    # Create indicator vector for this combination
    indicators = np.zeros((n, 8))
    indicators[:, idx_map[target_combination]] = 1
    
    # Exclude baseline indicator
    indicators_for_model = indicators[:, 1:]
    
    # Interactions
    interactions = []
    for i_comb in range(1, 8):
        for j_cov in range(n_covariates):
            interactions.append(X_centered[:, j_cov] * indicators[:, i_comb])
    interactions = np.column_stack(interactions) if interactions else np.zeros((n, 0))
    
    # Design matrix for prediction
    design_matrix = np.column_stack([indicators_for_model, X_centered, interactions])
    
    # Predict
    predictions = model.predict(design_matrix)
    
    return predictions


def aipw_estimator(Y, A, S, H, propensity_scores, X_covariates, target_combination, weights):
    """
    AIPW estimator with covariate adjustment.
    
    AIPW = 1/n * sum_i [(Y_i - Y_pred(a,s,h)) * I(A_i=a,S_i=s,H_i=h) / pi_i(a,s,h) * w_i] 
           + 1/n * sum_i [Y_pred(a,s,h) * w_i]
    
    Parameters:
    -----------
    Y : array-like
        Outcome variable
    A, S, H : array-like
        Treatment and spillover indicators
    propensity_scores : DataFrame
        First-order propensity scores
    X_covariates : DataFrame or array
        Covariates for outcome model
    target_combination : tuple
        Target (a, s, h) combination
    weights : array-like
        Weights gamma_i for each unit
    
    Returns:
    --------
    dict : Dictionary with AIPW estimate and components
    """
    Y = np.array(Y)
    A = np.array(A)
    S = np.array(S)
    H = np.array(H)
    weights = np.array(weights)
    
    a, s, h = target_combination
    n = len(Y)
    
    # Fit outcome model
    model, X_centered, design_matrix, idx_map = fit_outcome_model(Y, A, S, H, X_covariates)
    
    # Get predictions for target combination
    Y_pred = predict_outcome_model(model, X_centered, target_combination, idx_map)
    
    # Get propensity scores
    col_name = f'(A={a},S={s},H={h})'
    p_i = propensity_scores[col_name].values
    p_i = np.where(p_i > 0, p_i, 1e-10)
    
    # Create indicator
    indicator = ((A == a) & (S == s) & (H == h)).astype(float)
    
    # Compute AIPW estimate
    # First term: IPW with residuals
    residuals = Y - Y_pred
    ipw_term = np.mean(residuals * indicator * weights / p_i)
    
    # Second term: Mean of predictions
    pred_term = np.mean(Y_pred * weights)
    
    aipw_estimate = ipw_term + pred_term
    
    # For comparison, also compute IPW
    ipw_estimate = np.mean(Y * indicator * weights / p_i)
    
    return {
        'aipw': aipw_estimate,
        'ipw': ipw_estimate,
        'ipw_term': ipw_term,
        'pred_term': pred_term,
        'model': model,
        'Y_pred': Y_pred,
        'X_centered': X_centered,
        'idx_map': idx_map
    }


def compute_variance_aipw(Y, A, S, H, propensity_scores, X_covariates, target_combination, 
                          weights, village_idx_map, matrix_dir):
    """
    Compute variance estimate for AIPW estimator.
    
    Uses reweighted residuals: R_i = (Y_i - Y_pred(a,s,h)) * I(A_i=a,S_i=s,H_i=h) / pi_i(a,s,h)
    
    Parameters:
    -----------
    Y, A, S, H : array-like
        Outcome and treatment/spillover indicators
    propensity_scores : DataFrame
        First-order propensity scores
    X_covariates : DataFrame or array
        Covariates for outcome model
    target_combination : tuple
        Target (a, s, h) combination
    weights : array-like
        Weights gamma_i
    village_idx_map : dict
        Mapping from data index to village index
    matrix_dir : str
        Directory with second-order probability matrices
    
    Returns:
    --------
    dict : Dictionary with variance and SE
    """
    Y = np.array(Y)
    A = np.array(A)
    S = np.array(S)
    H = np.array(H)
    weights = np.array(weights)
    
    n = len(Y)
    a, s, h = target_combination
    
    # Fit outcome model and get predictions
    model, X_centered, design_matrix, idx_map = fit_outcome_model(Y, A, S, H, X_covariates)
    Y_pred = predict_outcome_model(model, X_centered, target_combination, idx_map)
    
    # Get propensity scores
    col_name = f'(A={a},S={s},H={h})'
    p_i = propensity_scores[col_name].values
    p_i = np.where(p_i > 0, p_i, 1e-10)
    
    # Load second-order probabilities
    p_ij_dict = load_second_order_probs(target_combination, matrix_dir)
    
    # Create indicator
    indicator = ((A == a) & (S == s) & (H == h)).astype(float)
    
    # Compute reweighted residuals: R_i = (Y_i - Y_pred) * indicator_i / p_i
    residuals = Y - Y_pred
    R_hat = residuals * indicator / p_i
    
    # AIPW Variance using residuals (same structure as IPW variance)
    # Diagonal terms
    diagonal_sum = 0.0
    for i in range(n):
        w_ii = 1 - p_i[i]
        diagonal_sum += w_ii * (R_hat[i] ** 2) * (weights[i] ** 2)
    
    # Off-diagonal terms
    offdiag_sum = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            vi = village_idx_map.get(i)
            vj = village_idx_map.get(j)
            
            if vi is None or vj is None:
                continue
            
            pair_key = (min(vi, vj), max(vi, vj))
            
            if pair_key in p_ij_dict:
                p_ij = p_ij_dict[pair_key]
                
                if p_ij >= 1e-4:
                    w_ij = (p_ij - p_i[i] * p_i[j]) / p_ij
                    offdiag_sum += 2 * w_ij * R_hat[i] * R_hat[j] * weights[i] * weights[j]
                else:
                    offdiag_sum += (p_i[i]/2 * R_hat[i]**2 * (weights[i]**2) + 
                                   p_i[j]/2 * R_hat[j]**2 * (weights[j]**2))
    
    aipw_variance = (diagonal_sum + offdiag_sum) / (n ** 2)
    
    return {
        'aipw_var': aipw_variance,
        'aipw_se': np.sqrt(aipw_variance) if aipw_variance > 0 else 0
    }


def aipw_hajek_estimator(Y, A, S, H, propensity_scores, X_covariates, target_combination, weights):
    """
    AIPW-Hajek estimator with covariate adjustment and Hajek-style normalization.
    
    AIPW-Hajek = ipw_hajek_term + pred_term
    where ipw_hajek_term = mean(residuals * indicator * weights / p_i) / [mean(indicator * weights / p_i) / mean(weights)]
    
    Parameters:
    -----------
    Y : array-like
        Outcome variable
    A, S, H : array-like
        Treatment and spillover indicators
    propensity_scores : DataFrame
        First-order propensity scores
    X_covariates : DataFrame or array
        Covariates for outcome model
    target_combination : tuple
        Target (a, s, h) combination
    weights : array-like
        Weights gamma_i for each unit
    
    Returns:
    --------
    dict : Dictionary with AIPW-Hajek estimate and components
    """
    Y = np.array(Y)
    A = np.array(A)
    S = np.array(S)
    H = np.array(H)
    weights = np.array(weights)
    
    a, s, h = target_combination
    n = len(Y)
    
    # Fit outcome model
    model, X_centered, design_matrix, idx_map = fit_outcome_model(Y, A, S, H, X_covariates)
    
    # Get predictions for target combination
    Y_pred = predict_outcome_model(model, X_centered, target_combination, idx_map)
    
    # Get propensity scores
    col_name = f'(A={a},S={s},H={h})'
    p_i = propensity_scores[col_name].values
    p_i = np.where(p_i > 0, p_i, 1e-10)
    
    # Create indicator
    indicator = ((A == a) & (S == s) & (H == h)).astype(float)
    
    # Compute AIPW-Hajek estimate
    # Residuals
    residuals = Y - Y_pred
    
    # IPW term with Hajek normalization
    ipw_numerator = np.mean(residuals * indicator * weights / p_i)
    hajek_denominator_numerator = np.mean(indicator * weights / p_i)
    mean_weights = np.mean(weights)
    hajek_denominator = hajek_denominator_numerator / mean_weights
    
    if hajek_denominator > 0:
        ipw_hajek_term = ipw_numerator / hajek_denominator
    else:
        ipw_hajek_term = 0
    
    # Prediction term
    pred_term = np.mean(Y_pred * weights)
    
    aipw_hajek_estimate = ipw_hajek_term + pred_term
    
    return {
        'aipw_hajek': aipw_hajek_estimate,
        'ipw_hajek_term': ipw_hajek_term,
        'pred_term': pred_term,
        'hajek_denominator': hajek_denominator,
        'model': model,
        'Y_pred': Y_pred,
        'X_centered': X_centered,
        'idx_map': idx_map
    }


def compute_variance_aipw_hajek(Y, A, S, H, propensity_scores, X_covariates, target_combination, 
                                weights, village_idx_map, matrix_dir):
    """
    Compute variance estimate for AIPW-Hajek estimator.
    
    Uses Hajek-demeaned residuals: R_i_hajek = Y_i - Y_pred(a,s,h) - aipw_hajek_estimate/mean_weights
    
    Parameters:
    -----------
    Y, A, S, H : array-like
        Outcome and treatment/spillover indicators
    propensity_scores : DataFrame
        First-order propensity scores
    X_covariates : DataFrame or array
        Covariates for outcome model
    target_combination : tuple
        Target (a, s, h) combination
    weights : array-like
        Weights gamma_i
    village_idx_map : dict
        Mapping from data index to village index
    matrix_dir : str
        Directory with second-order probability matrices
    
    Returns:
    --------
    dict : Dictionary with variance and SE
    """
    Y = np.array(Y)
    A = np.array(A)
    S = np.array(S)
    H = np.array(H)
    weights = np.array(weights)
    
    n = len(Y)
    a, s, h = target_combination
    
    # Get AIPW-Hajek estimate
    aipw_hajek_result = aipw_hajek_estimator(Y, A, S, H, propensity_scores, X_covariates, 
                                              target_combination, weights)
    aipw_hajek_est = aipw_hajek_result['aipw_hajek']
    Y_pred = aipw_hajek_result['Y_pred']
    ipw_hajek_term = aipw_hajek_result['ipw_hajek_term']
    mean_weights = np.mean(weights)
    
    # Get propensity scores
    col_name = f'(A={a},S={s},H={h})'
    p_i = propensity_scores[col_name].values
    p_i = np.where(p_i > 0, p_i, 1e-10)
    
    # Load second-order probabilities
    p_ij_dict = load_second_order_probs(target_combination, matrix_dir)
    
    # Create indicator
    indicator = ((A == a) & (S == s) & (H == h)).astype(float)
    
    # Compute Hajek-demeaned reweighted residuals
    # R_i_hajek = Y_i - Y_pred - aipw_hajek_estimate/mean_weights
    Y_demeaned = Y - Y_pred - ipw_hajek_term / mean_weights
    R_hat_hajek = Y_demeaned * indicator / p_i
    
    # Variance using Hajek-demeaned residuals
    # Diagonal terms
    diagonal_sum = 0.0
    for i in range(n):
        w_ii = 1 - p_i[i]
        diagonal_sum += w_ii * (R_hat_hajek[i] ** 2) * (weights[i] ** 2)
    
    # Off-diagonal terms
    offdiag_sum = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            vi = village_idx_map.get(i)
            vj = village_idx_map.get(j)
            
            if vi is None or vj is None:
                continue
            
            pair_key = (min(vi, vj), max(vi, vj))
            
            if pair_key in p_ij_dict:
                p_ij = p_ij_dict[pair_key]
                
                if p_ij >= 1e-4:
                    w_ij = (p_ij - p_i[i] * p_i[j]) / p_ij
                    offdiag_sum += 2 * w_ij * R_hat_hajek[i] * R_hat_hajek[j] * weights[i] * weights[j]
                else:
                    offdiag_sum += 0  # Conservative approach for very small p_ij
    
    aipw_hajek_variance = (diagonal_sum + offdiag_sum) / (n ** 2)
    
    return {
        'aipw_hajek_var': aipw_hajek_variance,
        'aipw_hajek_se': np.sqrt(aipw_hajek_variance) if aipw_hajek_variance > 0 else 0
    }


