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

# Set random seed for reproducibility
np.random.seed(42)

# Load data
print("Loading data...")
VillageLevel = pd.read_stata("/Users/leis/Documents/GitHub/Interference-CashTransfer/replication_materials/analysisdata/GE_VillageLevel_ECMA.dta")
GEO = pd.read_stata("/Users/leis/Documents/GitHub/Interference-CashTransfer/replication_materials/analysisdata/village_buffers_hhs_ge.dta")

print(f"VillageLevel shape: {VillageLevel.shape}")
print(f"GEO shape: {GEO.shape}")

# Get unique villages
villages = VillageLevel['village_code'].unique()
n_villages = len(villages)
print(f"Number of unique villages: {n_villages}")

# Create a mapping of village_code to sublocation_code and hi_sat
village_info = VillageLevel[['village_code', 'sublocation_code', 'hi_sat']].drop_duplicates()
village_info = village_info.set_index('village_code')

# Create a dictionary mapping each village to its nearby villages
# NEW LOGIC: Find at most 3 nearest villages with distance < 4 and NOT in same sublocation
print("\nFinding nearby villages (distance < 4, not in own sublocation, at most 3)...")
nearby_villages = {}

for village in villages:
    subloc = village_info.loc[village, 'sublocation_code']
    
    # Get all villages in GEO data for this village
    village_geo = GEO[GEO['treatvill_code'] == village].copy()
    
    # Filter to only villages NOT in the same sublocation and not the village itself
    village_geo = village_geo[village_geo['village_code'] != village]
    
    # Add sublocation info to filter out same sublocation
    village_geo['target_sublocation'] = village_geo['village_code'].map(
        lambda x: village_info.loc[x, 'sublocation_code'] if x in village_info.index else None
    )
    village_geo = village_geo[village_geo['target_sublocation'] != subloc]
    
    # NEW: Filter to only villages with distance < 4 (strictly less than)
    village_geo = village_geo[village_geo['distance'] < 4]
    
    # Sort by distance and take at most 3 closest
    village_geo = village_geo.sort_values('distance')
    nearest = village_geo['village_code'].head(3).tolist()
    
    nearby_villages[village] = nearest

# Print statistics about nearby villages
nearby_counts = [len(v) for v in nearby_villages.values()]
print(f"  Average number of nearby villages: {np.mean(nearby_counts):.2f}")
print(f"  Min: {np.min(nearby_counts)}, Max: {np.max(nearby_counts)}")
print(f"  Villages with 3 nearby: {sum([1 for c in nearby_counts if c == 3])}/{len(nearby_counts)}")
print(f"  Villages with 0 nearby: {sum([1 for c in nearby_counts if c == 0])}/{len(nearby_counts)}")

# Create a dictionary mapping each sublocation to its villages
sublocation_villages = {}
for village in villages:
    subloc = village_info.loc[village, 'sublocation_code']
    if subloc not in sublocation_villages:
        sublocation_villages[subloc] = []
    sublocation_villages[subloc].append(village)

print(f"Number of sublocations: {len(sublocation_villages)}")

# Map villages to indices (needed for dependent pairs construction)
village_to_idx = {village: idx for idx, village in enumerate(villages)}

# Build network of dependent pairs (neighbors and second-order neighbors)
print("\nBuilding network of dependent village pairs...")

# Find all neighbor pairs
dependent_pairs = set()

# Type 1: Villages in the same sublocation (direct neighbors for S)
for subloc, vills in sublocation_villages.items():
    for i in range(len(vills)):
        for j in range(i, len(vills)):  # Include i=j (self)
            v1, v2 = vills[i], vills[j]
            idx1, idx2 = village_to_idx.get(v1), village_to_idx.get(v2)
            if idx1 is not None and idx2 is not None:
                # Store as sorted tuple for consistency
                pair = (min(idx1, idx2), max(idx1, idx2))
                dependent_pairs.add(pair)

# Type 2: Villages in each other's nearby list (direct neighbors for H)
for village, near_list in nearby_villages.items():
    idx1 = village_to_idx[village]
    for near_village in near_list:
        if near_village in village_to_idx:
            idx2 = village_to_idx[near_village]
            pair = (min(idx1, idx2), max(idx1, idx2))
            dependent_pairs.add(pair)

# Type 3: Second-order neighbors (share a common neighbor)
print("  Finding second-order neighbors...")
# Build adjacency list
adjacency = {i: set() for i in range(n_villages)}

# Add sublocation neighbors
for subloc, vills in sublocation_villages.items():
    for v1 in vills:
        idx1 = village_to_idx[v1]
        for v2 in vills:
            if v1 != v2:
                idx2 = village_to_idx[v2]
                adjacency[idx1].add(idx2)

# Add nearby neighbors
for village, near_list in nearby_villages.items():
    idx1 = village_to_idx[village]
    for near_village in near_list:
        if near_village in village_to_idx:
            idx2 = village_to_idx[near_village]
            adjacency[idx1].add(idx2)
            adjacency[idx2].add(idx1)  # Symmetric

# Find second-order neighbors (villages that share at least one common neighbor)
for i in range(n_villages):
    neighbors_i = adjacency[i]
    for neighbor in neighbors_i:
        neighbors_of_neighbor = adjacency[neighbor]
        for j in neighbors_of_neighbor:
            if i != j:
                pair = (min(i, j), max(i, j))
                dependent_pairs.add(pair)

print(f"  Total dependent pairs: {len(dependent_pairs):,}")
print(f"  Percentage of all pairs: {len(dependent_pairs) / (n_villages * (n_villages + 1) / 2) * 100:.2f}%")
print(f"  Space savings: {(1 - len(dependent_pairs) / (n_villages * (n_villages + 1) / 2)) * 100:.2f}%")

# Monte Carlo simulation
n_simulations = 100000
print(f"\nRunning {n_simulations} Monte Carlo simulations...")
print("  (This will take a few minutes)")

# Initialize propensity score matrix: n_villages x 8
# Columns represent (A, S, H) combinations: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
propensity_scores = np.zeros((n_villages, 8))

# Initialize second-order inclusion probabilities (SPARSE - only for dependent pairs)
# Dictionary with 8 keys (one for each combination), each value is a dict of pair -> count
# P_ij(a,s,h) = probability that both village i and j are in combination (a,s,h)
# Only store for pairs in dependent_pairs
second_order_probs = {}
for a in [0, 1]:
    for s in [0, 1]:
        for h in [0, 1]:
            key = (a, s, h)
            second_order_probs[key] = {}

import time
start_time = time.time()

for sim in range(n_simulations):
    if (sim + 1) % 10000 == 0:
        elapsed = time.time() - start_time
        rate = (sim + 1) / elapsed
        remaining = (n_simulations - sim - 1) / rate
        print(f"  Simulation {sim + 1}/{n_simulations} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
    
    # Step 1: Randomize treatment assignment A for all villages
    A = {}
    for village in villages:
        hi_sat = village_info.loc[village, 'hi_sat']
        if hi_sat == 1:
            # High saturation: P(A=1) = 2/3
            A[village] = np.random.choice([0, 1], p=[1/3, 2/3])
        else:
            # Low saturation: P(A=1) = 1/3
            A[village] = np.random.choice([0, 1], p=[2/3, 1/3])
    
    # Step 2: Compute S for each village (more than half of other villages in same sublocation treated)
    S = {}
    for village in villages:
        subloc = village_info.loc[village, 'sublocation_code']
        other_villages_in_subloc = [v for v in sublocation_villages[subloc] if v != village]
        
        if len(other_villages_in_subloc) > 0:
            treated_count = sum([A[v] for v in other_villages_in_subloc])
            S[village] = 1 if treated_count > len(other_villages_in_subloc) / 2 else 0
        else:
            S[village] = 0
    
    # Step 3: Compute H for each village (more than half of nearby villages treated)
    # NOTE: If a village has 0 nearby villages (none within distance <= 6), H is always 0
    H = {}
    for village in villages:
        near_villages = nearby_villages[village]
        
        if len(near_villages) > 0:
            treated_count = sum([A[v] for v in near_villages])
            H[village] = 1 if treated_count > len(near_villages) / 2 else 0
        else:
            H[village] = 0
    
    # Step 4: Record (A, S, H) combination for each village
    # Also compute second-order inclusion probabilities (SPARSE - only dependent pairs)
    village_combinations = {}  # Map each village index to its (a,s,h) combination
    
    for village in villages:
        idx = village_to_idx[village]
        a, s, h = A[village], S[village], H[village]
        # Map (A, S, H) to column index: A*4 + S*2 + H
        col_idx = a * 4 + s * 2 + h
        propensity_scores[idx, col_idx] += 1
        
        # Store combination for this village
        village_combinations[idx] = (a, s, h)
    
    # Update second-order inclusion probabilities (only for dependent pairs)
    for pair in dependent_pairs:
        idx_i, idx_j = pair
        comb_i = village_combinations[idx_i]
        comb_j = village_combinations[idx_j]
        
        # If both villages are in the same combination, increment that combination's counter
        if comb_i == comb_j:
            combination = comb_i
            if pair not in second_order_probs[combination]:
                second_order_probs[combination][pair] = 0
            second_order_probs[combination][pair] += 1

# Convert counts to proportions
propensity_scores = propensity_scores / n_simulations

# Convert second-order probabilities to proportions (sparse format)
for key in second_order_probs:
    for pair in second_order_probs[key]:
        second_order_probs[key][pair] = second_order_probs[key][pair] / n_simulations

# Create DataFrame with propensity scores
column_names = [
    '(A=0,S=0,H=0)', '(A=0,S=0,H=1)', '(A=0,S=1,H=0)', '(A=0,S=1,H=1)',
    '(A=1,S=0,H=0)', '(A=1,S=0,H=1)', '(A=1,S=1,H=0)', '(A=1,S=1,H=1)'
]
propensity_df = pd.DataFrame(propensity_scores, columns=column_names)
propensity_df['village_code'] = villages

# Reorder columns to have village_code first
propensity_df = propensity_df[['village_code'] + column_names]

print("\nPropensity Score Matrix created!")
print(f"Shape: {propensity_df.shape}")
print("\nFirst few rows:")
print(propensity_df.head())

print("\nSummary statistics:")
print(propensity_df[column_names].describe())

# Verify that each row sums to 1
row_sums = propensity_df[column_names].sum(axis=1)
print(f"\nRow sums (should all be 1.0): min={row_sums.min():.6f}, max={row_sums.max():.6f}")

# Create output directory
import os
run_dir = "/Users/leis/Documents/GitHub/Interference-CashTransfer/analysis-2/runs/dist4_k3"
os.makedirs(run_dir, exist_ok=True)

# Save the propensity score matrix
output_path = f"{run_dir}/propensity_scores.csv"
propensity_df.to_csv(output_path, index=False)
print(f"\nPropensity scores saved to: {output_path}")

# Save second-order inclusion probability matrices (SPARSE FORMAT)
print("\n" + "=" * 80)
print("SECOND-ORDER INCLUSION PROBABILITY MATRICES (SPARSE)")
print("=" * 80)

# Create directory for matrices if it doesn't exist
matrix_dir = f"{run_dir}/second_order_matrices"
os.makedirs(matrix_dir, exist_ok=True)

for combination, sparse_probs in second_order_probs.items():
    a, s, h = combination
    print(f"\nCombination (A={a}, S={s}, H={h}):")
    print(f"  Number of stored pairs: {len(sparse_probs):,}")
    
    if len(sparse_probs) > 0:
        # Calculate statistics
        probs_list = list(sparse_probs.values())
        # Diagonal entries (where i==j)
        diagonal_probs = [prob for (i, j), prob in sparse_probs.items() if i == j]
        # Off-diagonal entries (where i!=j)
        off_diagonal_probs = [prob for (i, j), prob in sparse_probs.items() if i != j]
        
        if diagonal_probs:
            print(f"  Diagonal (P_ii) mean: {np.mean(diagonal_probs):.4f}, count: {len(diagonal_probs)}")
        if off_diagonal_probs:
            print(f"  Off-diagonal (P_ij, i≠j) mean: {np.mean(off_diagonal_probs):.6f}, count: {len(off_diagonal_probs):,}")
        print(f"  Min: {min(probs_list):.6f}, Max: {max(probs_list):.4f}")
        
        # Save as sparse format: village_i, village_j, probability
        sparse_data = []
        for (idx_i, idx_j), prob in sparse_probs.items():
            sparse_data.append({
                'village_i': villages[idx_i],
                'village_j': villages[idx_j],
                'village_i_idx': idx_i,
                'village_j_idx': idx_j,
                'probability': prob
            })
        
        sparse_df = pd.DataFrame(sparse_data)
        filename = f"{matrix_dir}/P_A{a}_S{s}_H{h}_sparse.csv"
        sparse_df.to_csv(filename, index=False)
        print(f"  Saved to: {filename}")
    else:
        print(f"  No pairs with this combination")

print("\n" + "=" * 80)
print("Summary of second-order matrices (SPARSE):")
print(f"  8 sparse matrices saved (one for each (A,S,H) combination)")
print(f"  Only {len(dependent_pairs):,} dependent pairs stored (out of {n_villages * (n_villages + 1) // 2:,} possible)")
print(f"  Space savings: {(1 - len(dependent_pairs) / (n_villages * (n_villages + 1) / 2)) * 100:.2f}%")
print(f"  P_ij(a,s,h) = probability that both village i and j are in combination (a,s,h)")
print(f"  Diagonal entries P_ii(a,s,h) = first-order propensity scores")
print(f"  For independent pairs (not in dependent_pairs), P_ij = P_i × P_j")
print("=" * 80)

print("\n" + "=" * 80)
print("ANALYSIS SPECIFICATION:")
print("  This analysis uses distance < 4 (strictly less than 4 km)")
print("  At most 3 nearest neighbors (outside own sublocation)")
print("  Villages with no neighbors meeting criteria will always have H=0")
print(f"  Results saved to: {run_dir}")
print("=" * 80)

