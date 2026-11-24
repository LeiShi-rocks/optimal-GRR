import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set_style("whitegrid")
sns.set_palette("viridis")

## Load data
print("Loading data...")
VillageLevel = pd.read_stata("/home/colligo/Interference-CashTransfer/replication_materials/analysisdata/GE_VillageLevel_ECMA.dta")
GEO = pd.read_stata("/home/colligo/Interference-CashTransfer/replication_materials/analysisdata/village_buffers_hhs_ge.dta")
propensity_df = pd.read_csv("/home/colligo/Interference-CashTransfer/analysis/eda/propensity_scores.csv")

# Import the functions from effect.py
import sys
sys.path.insert(0, '/home/colligo/Interference-CashTransfer/analysis/eda')
from effect import general_weighted_ipw_hajek, compute_variance_ipw_hajek

print("=" * 80)
print("ANALYZING MULTIPLE OUTCOMES")
print("=" * 80)

# Define outcomes to analyze
outcomes_dict = {
    'profit_per_ent': ('vill_profit1_wins', 'Profit per enterprise (KES)'),
    'revenue_per_ent': ('vill_revenue1_wins', 'Revenue per enterprise (KES)'),
    'cost_per_ent': ('vill_totcost_wins', 'Total costs per enterprise (KES)'),
    'wagebill_per_ent': ('vill_wagebill_wins', 'Wage bill per enterprise (KES)'),
}

# Prepare base data
village_info = VillageLevel[['village_code', 'sublocation_code', 'hi_sat', 'treat']].drop_duplicates()
village_info = village_info.set_index('village_code')

# Merge with propensity scores
data = propensity_df.merge(village_info, left_on='village_code', right_index=True)

# Get all outcomes
outcomes_data = VillageLevel[['village_code', 'n_allents'] + [v[0] for v in outcomes_dict.values()]].drop_duplicates(subset='village_code')

# Create per-enterprise versions
for key, (var, label) in outcomes_dict.items():
    outcomes_data[key] = outcomes_data[var] / outcomes_data['n_allents']

data = data.merge(outcomes_data, on='village_code')

# Reset index to ensure consecutive indices
data = data.reset_index(drop=True)

# Create village index mapping (data index -> village index in propensity_df)
village_idx_map = {}
for data_idx, row in data.iterrows():
    village_code = row['village_code']
    # Find this village's index in the original propensity_df
    village_idx = propensity_df[propensity_df['village_code'] == village_code].index[0]
    village_idx_map[data_idx] = village_idx

# Compute S (sublocation spillover)
print("\nComputing sublocation spillover indicator S...")
data['S'] = 0
for idx, row in data.iterrows():
    subloc = row['sublocation_code']
    village = row['village_code']
    other_villages = data[(data['sublocation_code'] == subloc) & (data['village_code'] != village)]
    if len(other_villages) > 0:
        treated_count = other_villages['treat'].sum()
        data.loc[idx, 'S'] = 1 if treated_count > len(other_villages) / 2 else 0

# Compute H (neighborhood spillover)
print("Computing neighborhood spillover indicator H...")
data['H'] = 0
for idx, row in data.iterrows():
    village = row['village_code']
    subloc = row['sublocation_code']
    
    village_geo = GEO[GEO['treatvill_code'] == village].copy()
    village_geo = village_geo[village_geo['village_code'] != village]
    
    village_geo['target_sublocation'] = village_geo['village_code'].map(
        lambda x: data[data['village_code'] == x]['sublocation_code'].iloc[0] if len(data[data['village_code'] == x]) > 0 else None
    )
    village_geo = village_geo[village_geo['target_sublocation'] != subloc]
    village_geo = village_geo.sort_values('distance')
    nearest_5 = village_geo['village_code'].head(5).tolist()
    
    if len(nearest_5) > 0:
        near_treatment = data[data['village_code'].isin(nearest_5)]['treat'].values
        treated_count = near_treatment.sum()
        data.loc[idx, 'H'] = 1 if treated_count > len(nearest_5) / 2 else 0

# Extract common variables
A = data['treat'].values
S = data['S'].values
H = data['H'].values
weights = np.ones(len(data))

ps_columns = [col for col in propensity_df.columns if col.startswith('(A=')]
propensity_scores = data[ps_columns]

print(f"\nData prepared: {len(data)} villages")
print(f"Treatment: {A.sum():.0f} treated, {len(A) - A.sum():.0f} control")
print(f"S: {S.sum():.0f} with sublocation spillover")
print(f"H: {H.sum():.0f} with neighborhood spillover")

# Run analysis for each outcome
all_results = []

for outcome_key, (outcome_var, outcome_label) in outcomes_dict.items():
    print(f"\n{'=' * 80}")
    print(f"OUTCOME: {outcome_label}")
    print(f"{'=' * 80}")
    
    Y = data[outcome_key].values
    
    print(f"Mean: {Y.mean():,.2f}, Median: {np.median(Y):,.2f}, Std: {Y.std():,.2f}")
    
    outcome_results = []
    
    for a in [0, 1]:
        for s in [0, 1]:
            for h in [0, 1]:
                target = (a, s, h)
                
                # Compute point estimates
                estimates = general_weighted_ipw_hajek(Y, A, S, H, propensity_scores, target, weights)
                
                # Compute variances
                print(f"  Computing variance for (A={a}, S={s}, H={h})...")
                variances = compute_variance_ipw_hajek(Y, A, S, H, propensity_scores, target, weights, village_idx_map)
                
                # Calculate 95% confidence intervals
                hajek_ci_lower = estimates['hajek'] - 1.96 * variances['hajek_se']
                hajek_ci_upper = estimates['hajek'] + 1.96 * variances['hajek_se']
                
                outcome_results.append({
                    'outcome': outcome_key,
                    'outcome_label': outcome_label,
                    'A': a, 'S': s, 'H': h,
                    'IPW': estimates['ipw'],
                    'IPW_SE': variances['ipw_se'],
                    'Hajek': estimates['hajek'],
                    'Hajek_SE': variances['hajek_se'],
                    'Hajek_CI_Lower': hajek_ci_lower,
                    'Hajek_CI_Upper': hajek_ci_upper,
                    'N': estimates['n_units_in_target'],
                    'Effective_N': estimates['effective_n']
                })
    
    # Display results for this outcome
    outcome_df = pd.DataFrame(outcome_results)
    print("\nHajek Estimates by (A, S, H) with Standard Errors:")
    print(outcome_df[['A', 'S', 'H', 'Hajek', 'Hajek_SE', 'Hajek_CI_Lower', 'Hajek_CI_Upper']].to_string(index=False))
    
    all_results.extend(outcome_results)

# Save all results
all_results_df = pd.DataFrame(all_results)
output_path = "/home/colligo/Interference-CashTransfer/analysis/eda/multiple_outcomes_with_variance.csv"
all_results_df.to_csv(output_path, index=False)
print(f"\n{'=' * 80}")
print(f"All results saved to: {output_path}")
print(f"{'=' * 80}")

# Create comparison table for key treatment effects
print("\n" + "=" * 80)
print("TREATMENT EFFECTS COMPARISON (Hajek Estimates)")
print("=" * 80)

comparison_data = []
for outcome_key, (_, outcome_label) in outcomes_dict.items():
    outcome_subset = all_results_df[all_results_df['outcome'] == outcome_key]
    
    # Direct effect (S=0, H=0)
    control_000_est = outcome_subset[(outcome_subset['A']==0) & (outcome_subset['S']==0) & (outcome_subset['H']==0)]['Hajek'].values[0]
    control_000_se = outcome_subset[(outcome_subset['A']==0) & (outcome_subset['S']==0) & (outcome_subset['H']==0)]['Hajek_SE'].values[0]
    treat_100_est = outcome_subset[(outcome_subset['A']==1) & (outcome_subset['S']==0) & (outcome_subset['H']==0)]['Hajek'].values[0]
    treat_100_se = outcome_subset[(outcome_subset['A']==1) & (outcome_subset['S']==0) & (outcome_subset['H']==0)]['Hajek_SE'].values[0]
    effect_000 = treat_100_est - control_000_est
    effect_000_se = np.sqrt(control_000_se**2 + treat_100_se**2)  # Assuming independence
    pct_000 = (effect_000 / control_000_est * 100) if control_000_est != 0 else 0
    
    # With sublocation spillover (S=1, H=0)
    control_010_est = outcome_subset[(outcome_subset['A']==0) & (outcome_subset['S']==1) & (outcome_subset['H']==0)]['Hajek'].values[0]
    control_010_se = outcome_subset[(outcome_subset['A']==0) & (outcome_subset['S']==1) & (outcome_subset['H']==0)]['Hajek_SE'].values[0]
    treat_110_est = outcome_subset[(outcome_subset['A']==1) & (outcome_subset['S']==1) & (outcome_subset['H']==0)]['Hajek'].values[0]
    treat_110_se = outcome_subset[(outcome_subset['A']==1) & (outcome_subset['S']==1) & (outcome_subset['H']==0)]['Hajek_SE'].values[0]
    effect_010 = treat_110_est - control_010_est
    effect_010_se = np.sqrt(control_010_se**2 + treat_110_se**2)
    pct_010 = (effect_010 / control_010_est * 100) if control_010_est != 0 else 0
    
    # With both spillovers (S=1, H=1)
    control_011_est = outcome_subset[(outcome_subset['A']==0) & (outcome_subset['S']==1) & (outcome_subset['H']==1)]['Hajek'].values[0]
    control_011_se = outcome_subset[(outcome_subset['A']==0) & (outcome_subset['S']==1) & (outcome_subset['H']==1)]['Hajek_SE'].values[0]
    treat_111_est = outcome_subset[(outcome_subset['A']==1) & (outcome_subset['S']==1) & (outcome_subset['H']==1)]['Hajek'].values[0]
    treat_111_se = outcome_subset[(outcome_subset['A']==1) & (outcome_subset['S']==1) & (outcome_subset['H']==1)]['Hajek_SE'].values[0]
    effect_011 = treat_111_est - control_011_est
    effect_011_se = np.sqrt(control_011_se**2 + treat_111_se**2)
    pct_011 = (effect_011 / control_011_est * 100) if control_011_est != 0 else 0
    
    # Calculate t-statistics and significance
    t_stat_000 = effect_000 / effect_000_se if effect_000_se > 0 else 0
    t_stat_010 = effect_010 / effect_010_se if effect_010_se > 0 else 0
    t_stat_011 = effect_011 / effect_011_se if effect_011_se > 0 else 0
    
    # Significance indicators (|t| > 1.96 for 95% CI)
    sig_000 = '***' if abs(t_stat_000) > 2.576 else ('**' if abs(t_stat_000) > 1.96 else ('*' if abs(t_stat_000) > 1.645 else ''))
    sig_010 = '***' if abs(t_stat_010) > 2.576 else ('**' if abs(t_stat_010) > 1.96 else ('*' if abs(t_stat_010) > 1.645 else ''))
    sig_011 = '***' if abs(t_stat_011) > 2.576 else ('**' if abs(t_stat_011) > 1.96 else ('*' if abs(t_stat_011) > 1.645 else ''))
    
    comparison_data.append({
        'Outcome': outcome_label.replace(' per enterprise (KES)', ''),
        'Direct_Effect': effect_000,
        'Direct_SE': effect_000_se,
        'Direct_Sig': sig_000,
        'Subloc_Effect': effect_010,
        'Subloc_SE': effect_010_se,
        'Subloc_Sig': sig_010,
        'Both_Effect': effect_011,
        'Both_SE': effect_011_se,
        'Both_Sig': sig_011
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nTreatment Effects (Treated - Control) with Standard Errors:")
print(comparison_df.to_string(index=False))

print("\n" + "=" * 80)
print("Note: All values are per enterprise. Effects in KES (~115 KES = 1 USD)")
print("Significance: *** p<0.01, ** p<0.05, * p<0.10")
print("=" * 80)

