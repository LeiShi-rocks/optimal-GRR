"""
Compute propensity scores and second-order inclusion probabilities for GRR analysis.

This script uses Monte Carlo simulation to estimate:
1. First-order propensity scores: P_i(A, H) for each village
2. Second-order inclusion probabilities: P_ij(A, H) for dependent pairs

The two-stage randomization design:
- Stage 1: Sublocations assigned to high-saturation (hi_sat=1) or low-saturation (hi_sat=0)
- Stage 2: Within sublocations:
    - High saturation: P(A=1) = 2/3
    - Low saturation: P(A=1) = 1/3

Exposure mapping:
- A: self treatment (0 or 1)
- H: whether more than 1/3 of connected villages are treated (0 or 1)
- D = A * 2 + H gives 4 levels: (0,0), (0,1), (1,0), (1,1)

Output files:
- propensity_scores.csv: First-order propensity scores
- second_order_matrices/P_A{a}_H{h}_sparse.csv: Second-order matrices (sparse)
- inclusion_prob_info.json: Metadata

Usage:
    python compute_inclusion_prob.py [--n_simulations 100000]
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import sys
import json
import time
import argparse
from datetime import datetime
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DERIVED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "data", "cash-transfer-derived"))
PROCESSED_DIR = os.path.join(DERIVED_DIR, "processed_data")
OUTPUT_DIR = os.path.join(DERIVED_DIR, "inclusion_probabilities")

# Default Monte Carlo settings
DEFAULT_N_SIMULATIONS = 100000


# =============================================================================
# Helper Functions
# =============================================================================

def load_processed_data():
    """Load preprocessed village data and adjacency matrix."""
    print("Loading processed data...")
    
    village_data_path = os.path.join(PROCESSED_DIR, "village_data.csv")
    adj_path = os.path.join(PROCESSED_DIR, "adjacency_matrix.npy")
    
    if not os.path.exists(village_data_path):
        raise FileNotFoundError(
            f"Village data not found: {village_data_path}\n"
            "Please run preprocess.py first."
        )
    
    if not os.path.exists(adj_path):
        raise FileNotFoundError(
            f"Adjacency matrix not found: {adj_path}\n"
            "Please run preprocess.py first."
        )
    
    village_data = pd.read_csv(village_data_path)
    G = np.load(adj_path)
    
    print(f"  Loaded {len(village_data)} villages")
    print(f"  Adjacency matrix shape: {G.shape}")
    
    return village_data, G


def build_dependent_pairs(village_data, G):
    """
    Build set of dependent village pairs.
    
    Two villages are dependent if:
    1. They are geographic neighbors (direct edge in G)
    2. They share a common neighbor (second-order dependency)
    
    For independent pairs, we can compute P_ij = P_i * P_j.
    """
    print("\nBuilding network of dependent village pairs...")
    
    n = len(village_data)
    
    # Build adjacency list from G (excluding self-loops)
    adjacency = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and G[i, j] > 0:
                adjacency[i].add(j)
    
    dependent_pairs = set()
    
    # Type 1: Self-pairs (diagonal)
    for i in range(n):
        dependent_pairs.add((i, i))
    
    # Type 2: Direct neighbors (edges in G)
    for i in range(n):
        for j in adjacency[i]:
            if i <= j:  # Avoid duplicates
                dependent_pairs.add((i, j))
    
    # Type 3: Second-order neighbors (share a common neighbor)
    print("  Finding second-order neighbors...")
    for i in range(n):
        neighbors_i = adjacency[i]
        for neighbor in neighbors_i:
            neighbors_of_neighbor = adjacency[neighbor]
            for j in neighbors_of_neighbor:
                if i <= j:
                    dependent_pairs.add((i, j))
    
    # Type 4: Same sublocation (treatment is correlated through hi_sat)
    # Villages in the same sublocation share the same saturation assignment
    subloc_to_villages = {}
    for idx, row in village_data.iterrows():
        subloc = row['sublocation_code']
        if subloc not in subloc_to_villages:
            subloc_to_villages[subloc] = []
        subloc_to_villages[subloc].append(idx)
    
    print("  Adding same-sublocation pairs...")
    for subloc, villages in subloc_to_villages.items():
        for i in range(len(villages)):
            for j in range(i, len(villages)):
                dependent_pairs.add((villages[i], villages[j]))
    
    print(f"  Total dependent pairs: {len(dependent_pairs):,}")
    print(f"  Percentage of all pairs: {len(dependent_pairs) / (n * (n + 1) / 2) * 100:.2f}%")
    print(f"  Space savings: {(1 - len(dependent_pairs) / (n * (n + 1) / 2)) * 100:.2f}%")
    
    return dependent_pairs, adjacency


def simulate_treatment(village_data):
    """
    Simulate one draw from the treatment assignment mechanism.
    
    Two-stage design:
    - High saturation (hi_sat=1): P(A=1) = 2/3
    - Low saturation (hi_sat=0): P(A=1) = 1/3
    """
    n = len(village_data)
    A = np.zeros(n, dtype=int)
    
    for idx, row in village_data.iterrows():
        hi_sat = row['hi_sat']
        if hi_sat == 1:
            A[idx] = np.random.choice([0, 1], p=[1/3, 2/3])
        else:
            A[idx] = np.random.choice([0, 1], p=[2/3, 1/3])
    
    return A


def compute_exposure(A, adjacency):
    """
    Compute exposure mapping H given treatment A.
    
    H_i = 1 if more than 1/3 of neighboring villages are treated
    """
    n = len(A)
    H = np.zeros(n, dtype=int)
    
    for i in range(n):
        neighbors = list(adjacency[i])
        
        if len(neighbors) > 0:
            treated_count = A[neighbors].sum()
            H[i] = 1 if treated_count > len(neighbors) / 3 else 0  # More than 1/3 treated
    
    return H


def run_monte_carlo(village_data, adjacency, dependent_pairs, n_simulations):
    """
    Run Monte Carlo simulation to estimate propensity scores.
    
    Exposure levels (k=4):
    - d=0: (A=0, H=0)
    - d=1: (A=0, H=1)
    - d=2: (A=1, H=0)
    - d=3: (A=1, H=1)
    """
    print("\n" + "=" * 80)
    print(f"RUNNING MONTE CARLO SIMULATION ({n_simulations:,} iterations)")
    print("=" * 80)
    
    n = len(village_data)
    k = 4  # Number of exposure levels
    
    # Initialize propensity score matrix: n x 4
    # Columns: (A=0,H=0), (A=0,H=1), (A=1,H=0), (A=1,H=1)
    propensity_counts = np.zeros((n, k), dtype=np.int64)
    
    # Initialize second-order inclusion probability counts (sparse)
    # Key: (a, h), Value: dict of (i, j) -> count
    second_order_counts = {(a, h): {} for a in [0, 1] for h in [0, 1]}
    
    start_time = time.time()
    
    for sim in tqdm(range(n_simulations), desc="Monte Carlo simulations", 
                    unit="sim", ncols=80):
        # Simulate treatment assignment
        A = simulate_treatment(village_data)
        
        # Compute exposure
        H = compute_exposure(A, adjacency)
        
        # Record (A, H) combination for each village
        village_combinations = {}
        for i in range(n):
            a, h = A[i], H[i]
            col_idx = a * 2 + h  # Map (A, H) to column index
            propensity_counts[i, col_idx] += 1
            village_combinations[i] = (a, h)
        
        # Update second-order counts (only for dependent pairs)
        for pair in dependent_pairs:
            i, j = pair
            comb_i = village_combinations[i]
            comb_j = village_combinations[j]
            
            # If both villages have the same combination, increment
            if comb_i == comb_j:
                if pair not in second_order_counts[comb_i]:
                    second_order_counts[comb_i][pair] = 0
                second_order_counts[comb_i][pair] += 1
    
    # Convert counts to probabilities
    propensity_scores = propensity_counts / n_simulations
    
    second_order_probs = {}
    for key in second_order_counts:
        second_order_probs[key] = {}
        for pair, count in second_order_counts[key].items():
            second_order_probs[key][pair] = count / n_simulations
    
    elapsed = time.time() - start_time
    print(f"\nSimulation complete in {elapsed:.1f}s")
    
    return propensity_scores, second_order_probs


def run_monte_carlo_vectorized(village_data, adjacency, dependent_pairs, n_simulations, seed, batch=2000):
    """Same design, exposure rule and outputs as run_monte_carlo, simulated in batches with matrix products.

    The random stream differs from the loop version (np.random.default_rng instead of per-village
    np.random.choice), so results agree up to Monte Carlo error.
    """
    n = len(village_data)
    rng = np.random.default_rng(seed)
    Adj = np.zeros((n, n))
    for i, nb in adjacency.items():
        Adj[i, list(nb)] = 1.0
    deg = Adj.sum(1)
    p = np.where(village_data['hi_sat'].values == 1, 2 / 3, 1 / 3)
    counts = np.zeros((n, 4))
    joint = np.zeros((4, n, n))
    done = 0
    while done < n_simulations:
        b = min(batch, n_simulations - done)
        A = (rng.random((b, n)) < p).astype(float)
        H = ((A @ Adj) > deg / 3) & (deg > 0)
        Dm = (2 * A + H).astype(int)
        for d in range(4):
            M = (Dm == d).astype(float)
            counts[:, d] += M.sum(0)
            joint[d] += M.T @ M
        done += b
        print(f"  {done:,}/{n_simulations:,} draws", flush=True)
    second_order_probs = {}
    for d in range(4):
        a, h = d // 2, d % 2
        second_order_probs[(a, h)] = {(i, j): joint[d, i, j] / n_simulations
                                      for (i, j) in dependent_pairs if joint[d, i, j] > 0}
    return counts / n_simulations, second_order_probs


def save_results(village_data, propensity_scores, second_order_probs, dependent_pairs, n_simulations, output_dir):
    """Save propensity scores and second-order matrices."""
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save propensity scores
    column_names = [
        '(A=0,H=0)', '(A=0,H=1)', '(A=1,H=0)', '(A=1,H=1)'
    ]
    propensity_df = pd.DataFrame(propensity_scores, columns=column_names)
    propensity_df.insert(0, 'village_code', village_data['village_code'].values)
    
    propensity_path = os.path.join(output_dir, "propensity_scores.csv")
    propensity_df.to_csv(propensity_path, index=False)
    print(f"Saved propensity scores to: {propensity_path}")
    
    # Verify row sums
    row_sums = propensity_df[column_names].sum(axis=1)
    print(f"  Row sums (should be ~1.0): min={row_sums.min():.6f}, max={row_sums.max():.6f}")
    
    # Print propensity score summaries
    print("\nPropensity score summaries:")
    for col in column_names:
        vals = propensity_df[col]
        print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")
    
    # 2. Save second-order matrices (sparse format)
    matrix_dir = os.path.join(output_dir, "second_order_matrices")
    os.makedirs(matrix_dir, exist_ok=True)
    
    print("\nSecond-order inclusion probability matrices:")
    for (a, h), sparse_probs in second_order_probs.items():
        print(f"\n  Combination (A={a}, H={h}):")
        print(f"    Number of stored pairs: {len(sparse_probs):,}")
        
        if len(sparse_probs) > 0:
            probs_list = list(sparse_probs.values())
            diagonal_probs = [prob for (i, j), prob in sparse_probs.items() if i == j]
            off_diagonal_probs = [prob for (i, j), prob in sparse_probs.items() if i != j]
            
            if diagonal_probs:
                print(f"    Diagonal (P_ii) mean: {np.mean(diagonal_probs):.4f}, count: {len(diagonal_probs)}")
            if off_diagonal_probs:
                print(f"    Off-diagonal (P_ij) mean: {np.mean(off_diagonal_probs):.6f}, count: {len(off_diagonal_probs):,}")
            print(f"    Min: {min(probs_list):.6f}, Max: {max(probs_list):.4f}")
            
            # Save as sparse format
            sparse_data = []
            for (idx_i, idx_j), prob in sparse_probs.items():
                sparse_data.append({
                    'village_i': village_data.iloc[idx_i]['village_code'],
                    'village_j': village_data.iloc[idx_j]['village_code'],
                    'village_i_idx': idx_i,
                    'village_j_idx': idx_j,
                    'probability': prob
                })
            
            sparse_df = pd.DataFrame(sparse_data)
            filename = os.path.join(matrix_dir, f"P_A{a}_H{h}_sparse.csv")
            sparse_df.to_csv(filename, index=False)
            print(f"    Saved to: {filename}")
        else:
            print(f"    No pairs with this combination")
    
    # 3. Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_simulations': n_simulations,
        'n_villages': len(village_data),
        'n_dependent_pairs': len(dependent_pairs),
        'exposure_mapping': {
            'description': '(A, H) where A is self-treatment, H is more than 1/3 neighbors treated',
            'n_levels': 4,
            'encoding': 'D = A * 2 + H',
        },
        'treatment_mechanism': {
            'high_saturation_prob': 2/3,
            'low_saturation_prob': 1/3,
        },
        'propensity_summary': {
            col: {
                'mean': float(propensity_df[col].mean()),
                'std': float(propensity_df[col].std()),
                'min': float(propensity_df[col].min()),
                'max': float(propensity_df[col].max())
            }
            for col in column_names
        }
    }
    
    metadata_path = os.path.join(output_dir, "inclusion_prob_info.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute propensity scores via Monte Carlo')
    parser.add_argument('--n_simulations', type=int, default=DEFAULT_N_SIMULATIONS,
                       help=f'Number of Monte Carlo simulations (default: {DEFAULT_N_SIMULATIONS})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--loop', action='store_true',
                       help='Use the original per-draw Python loop (about 6 h for 100,000 draws)')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("COMPUTING INCLUSION PROBABILITIES FOR GRR ANALYSIS")
    print("Exposure mapping: (A, H) - 4 levels")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Number of simulations: {args.n_simulations:,}")
    print(f"Random seed: {args.seed}")
    
    np.random.seed(args.seed)
    
    try:
        # Load processed data
        village_data, G = load_processed_data()
        
        # Build dependent pairs
        dependent_pairs, adjacency = build_dependent_pairs(village_data, G)
        
        # Run Monte Carlo simulation
        if args.loop:
            propensity_scores, second_order_probs = run_monte_carlo(
                village_data, adjacency, dependent_pairs, args.n_simulations
            )
        else:
            propensity_scores, second_order_probs = run_monte_carlo_vectorized(
                village_data, adjacency, dependent_pairs, args.n_simulations, args.seed
            )
        
        # Save results
        save_results(village_data, propensity_scores, second_order_probs, 
                    dependent_pairs, args.n_simulations, OUTPUT_DIR)
        
        print("\n" + "=" * 80)
        print("COMPUTATION COMPLETE!")
        print("=" * 80)
        print(f"Output directory: {OUTPUT_DIR}")
        print("\nNext step: Run GRR-analysis.py to estimate treatment effects")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
