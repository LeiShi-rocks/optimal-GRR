import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Set the style of the plots
sns.set_style("whitegrid")

# Set the color palette
sns.set_palette("viridis")

## Load data
VillageLevel = pd.read_stata("/home/colligo/Interference-CashTransfer/replication_materials/analysisdata/GE_VillageLevel_ECMA.dta")
GEO = pd.read_stata("/home/colligo/Interference-CashTransfer/replication_materials/analysisdata/village_buffers_hhs_ge.dta")

## Load propensity scores
propensity_df = pd.read_csv("/home/colligo/Interference-CashTransfer/analysis/eda/propensity_scores.csv")

## Load second-order inclusion probabilities (sparse format)
def load_second_order_probs(combination):
    """Load sparse second-order inclusion probability matrix for a given combination."""
    a, s, h = combination
    filename = f"/home/colligo/Interference-CashTransfer/analysis/eda/second_order_matrices/P_A{a}_S{s}_H{h}_sparse.csv"
    sparse_df = pd.read_csv(filename)
    # Create dictionary: (idx_i, idx_j) -> probability
    prob_dict = {}
    for _, row in sparse_df.iterrows():
        idx_i = int(row['village_i_idx'])
        idx_j = int(row['village_j_idx'])
        prob = row['probability']
        prob_dict[(idx_i, idx_j)] = prob
    return prob_dict

## Implementation of a general weighted IPW/Hajek estimator
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
        H = 1 if more than half of the 5 closest villages OUTSIDE own sublocation are treated
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
    
    Notes:
    ------
    IPW estimator = mean(Y * indicator{A,S,H = (a,s,h)} * gamma_i / propensity_score(a,s,h))
    
    Hajek estimator = IPW_numerator / Hajek_denominator
    where Hajek_denominator = mean(indicator{A,S,H = (a,s,h)} * gamma_i / propensity_score(a,s,h)) / mean(gamma_i)
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


def compute_variance_ipw_hajek(Y, A, S, H, propensity_scores, target_combination, weights, village_idx_map):
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
    
    Returns:
    --------
    dict : Dictionary with 'ipw_var' and 'hajek_var'
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
    p_ij_dict = load_second_order_probs(target_combination)
    
    # Create indicator
    indicator = ((A == a) & (S == s) & (H == h)).astype(float)
    
    # Compute IPW reweighted outcomes: Yhat_i = Y_i * indicator_i / p_i
    Yhat = Y * indicator / p_i
    
    # IPW Variance
    # Diagonal terms: w_ii = (1 - p_ii), sum w_ii * Yhat_i^2
    diagonal_sum = 0.0
    for i in range(n):
        w_ii = 1 - p_i[i]
        diagonal_sum += w_ii * (Yhat[i] ** 2)
    
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
                    offdiag_sum += 2 * w_ij * Yhat[i] * Yhat[j]  # 2x for symmetry
                else:
                    # Conservative estimate for very small p_ij
                    offdiag_sum += (p_i[i]/2 * Yhat[i]**2 + p_i[j]/2 * Yhat[j]**2)
    
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
        diagonal_sum_hajek += w_ii * (Yhat_demean[i] ** 2)
    
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
                    offdiag_sum_hajek += 2 * w_ij * Yhat_demean[i] * Yhat_demean[j]
                else:
                    offdiag_sum_hajek += (p_i[i]/2 * Yhat_demean[i]**2 + p_i[j]/2 * Yhat_demean[j]**2)
    
    hajek_variance = (diagonal_sum_hajek + offdiag_sum_hajek) / (n ** 2)
    
    return {
        'ipw_var': ipw_variance,
        'hajek_var': hajek_variance,
        'ipw_se': np.sqrt(ipw_variance) if ipw_variance > 0 else 0,
        'hajek_se': np.sqrt(hajek_variance) if hajek_variance > 0 else 0
    }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EXAMPLE: Computing IPW and Hajek Estimates")
    print("=" * 80)
    
    # Prepare data: get actual treatment assignment from VillageLevel
    # We need to compute actual S and H for each village based on observed treatment
    
    # Get unique villages and their treatment status
    village_info = VillageLevel[['village_code', 'sublocation_code', 'hi_sat', 'treat']].drop_duplicates()
    village_info = village_info.set_index('village_code')
    
    # Merge with propensity scores
    data = propensity_df.merge(village_info, left_on='village_code', right_index=True)
    
    # Create village index mapping (data index -> village index in propensity_df)
    village_idx_map = {}
    for data_idx, row in data.iterrows():
        village_code = row['village_code']
        # Find this village's index in the original propensity_df
        village_idx = propensity_df[propensity_df['village_code'] == village_code].index[0]
        village_idx_map[data_idx] = village_idx
    
    # For demonstration, let's use profit per enterprise
    # Get outcome from VillageLevel (first occurrence of each village)
    outcomes = VillageLevel[['village_code', 'vill_profit1_wins', 'n_allents']].drop_duplicates(subset='village_code')
    outcomes['profit_per_enterprise'] = outcomes['vill_profit1_wins'] / outcomes['n_allents']
    data = data.merge(outcomes, on='village_code')
    
    # Compute S (sublocation spillover) based on actual treatment
    # S = 1 if more than half of other villages in same sublocation are treated
    data['S'] = 0
    for idx, row in data.iterrows():
        subloc = row['sublocation_code']
        village = row['village_code']
        # Get other villages in same sublocation
        other_villages = data[(data['sublocation_code'] == subloc) & (data['village_code'] != village)]
        if len(other_villages) > 0:
            treated_count = other_villages['treat'].sum()
            data.loc[idx, 'S'] = 1 if treated_count > len(other_villages) / 2 else 0
    
    # Compute H (neighborhood spillover) based on actual treatment
    # H = 1 if more than half of the 5 closest villages OUTSIDE own sublocation are treated
    print("\nComputing neighborhood spillover indicator H...")
    print("  (Finding 5 closest villages outside own sublocation for each village)")
    
    data['H'] = 0
    for idx, row in data.iterrows():
        village = row['village_code']
        subloc = row['sublocation_code']
        
        # Get all villages in GEO data for this village
        village_geo = GEO[GEO['treatvill_code'] == village].copy()
        
        # Filter to only villages NOT in the same sublocation and not the village itself
        village_geo = village_geo[village_geo['village_code'] != village]
        
        # Add sublocation info to filter out same sublocation
        village_geo['target_sublocation'] = village_geo['village_code'].map(
            lambda x: data[data['village_code'] == x]['sublocation_code'].iloc[0] if len(data[data['village_code'] == x]) > 0 else None
        )
        village_geo = village_geo[village_geo['target_sublocation'] != subloc]
        
        # Sort by distance and take the 5 closest
        village_geo = village_geo.sort_values('distance')
        nearest_5 = village_geo['village_code'].head(5).tolist()
        
        if len(nearest_5) > 0:
            # Get treatment status of these 5 nearest villages
            near_treatment = data[data['village_code'].isin(nearest_5)]['treat'].values
            treated_count = near_treatment.sum()
            data.loc[idx, 'H'] = 1 if treated_count > len(nearest_5) / 2 else 0
        else:
            data.loc[idx, 'H'] = 0
    
    # Extract variables
    Y = data['profit_per_enterprise'].values
    A = data['treat'].values
    S = data['S'].values
    H = data['H'].values
    
    # Use uniform weights for now (can be modified)
    weights = np.ones(len(data))
    
    # Select propensity score columns
    ps_columns = [col for col in propensity_df.columns if col.startswith('(A=')]
    propensity_scores = data[ps_columns]
    
    print(f"\nData prepared:")
    print(f"  Number of villages: {len(data)}")
    print(f"  Outcome variable: profit_per_enterprise (vill_profit1_wins / n_allents)")
    print(f"  Outcome mean: {Y.mean():,.2f} KES, median: {np.median(Y):,.2f} KES")
    print(f"  Treatment distribution: {A.sum():.0f} treated, {len(A) - A.sum():.0f} control")
    print(f"  S distribution: {S.sum():.0f} with sublocation spillover")
    print(f"  H distribution: {H.sum():.0f} with neighborhood spillover")
    
    # Compute estimates for all 8 combinations
    print("\n" + "=" * 80)
    print("ESTIMATES FOR ALL (A, S, H) COMBINATIONS (WITH VARIANCE)")
    print("=" * 80)
    
    # Reset index to ensure consecutive indices for village_idx_map
    data = data.reset_index(drop=True)
    # Rebuild village_idx_map with new indices
    village_idx_map = {}
    for data_idx, row in data.iterrows():
        village_code = row['village_code']
        village_idx = propensity_df[propensity_df['village_code'] == village_code].index[0]
        village_idx_map[data_idx] = village_idx
    
    # Recompute variables with new indices
    Y = data['profit_per_enterprise'].values
    A = data['treat'].values
    S = data['S'].values
    H = data['H'].values
    ps_columns = [col for col in propensity_df.columns if col.startswith('(A=')]
    propensity_scores = data[ps_columns]
    
    results = []
    for a in [0, 1]:
        for s in [0, 1]:
            for h in [0, 1]:
                target = (a, s, h)
                
                # Compute point estimates
                estimates = general_weighted_ipw_hajek(Y, A, S, H, propensity_scores, target, weights)
                
                # Compute variances
                print(f"\nComputing variance for (A={a}, S={s}, H={h})...")
                variances = compute_variance_ipw_hajek(Y, A, S, H, propensity_scores, target, weights, village_idx_map)
                
                # Calculate 95% confidence intervals
                ipw_ci_lower = estimates['ipw'] - 1.96 * variances['ipw_se']
                ipw_ci_upper = estimates['ipw'] + 1.96 * variances['ipw_se']
                hajek_ci_lower = estimates['hajek'] - 1.96 * variances['hajek_se']
                hajek_ci_upper = estimates['hajek'] + 1.96 * variances['hajek_se']
                
                print(f"  IPW estimate: {estimates['ipw']:,.2f} (SE: {variances['ipw_se']:,.2f})")
                print(f"  IPW 95% CI: [{ipw_ci_lower:,.2f}, {ipw_ci_upper:,.2f}]")
                print(f"  Hajek estimate: {estimates['hajek']:,.2f} (SE: {variances['hajek_se']:,.2f})")
                print(f"  Hajek 95% CI: [{hajek_ci_lower:,.2f}, {hajek_ci_upper:,.2f}]")
                print(f"  Number of units: {estimates['n_units_in_target']}")
                print(f"  Effective N: {estimates['effective_n']:.2f}")
                
                results.append({
                    'A': a, 'S': s, 'H': h,
                    'IPW': estimates['ipw'],
                    'IPW_SE': variances['ipw_se'],
                    'IPW_CI_Lower': ipw_ci_lower,
                    'IPW_CI_Upper': ipw_ci_upper,
                    'Hajek': estimates['hajek'],
                    'Hajek_SE': variances['hajek_se'],
                    'Hajek_CI_Lower': hajek_ci_lower,
                    'Hajek_CI_Upper': hajek_ci_upper,
                    'N': estimates['n_units_in_target'],
                    'Effective_N': estimates['effective_n']
                })
    
    # Create summary table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Save results
    output_path = "/home/colligo/Interference-CashTransfer/analysis/eda/treatment_effect_estimates_with_variance.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"\nNote: Estimates are in KES (Kenyan Shillings) per enterprise")
    print(f"      Approximate conversion: 1 USD ≈ 115 KES")
    print(f"      Standard errors and 95% confidence intervals included")
