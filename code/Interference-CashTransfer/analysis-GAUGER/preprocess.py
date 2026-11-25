"""
Preprocess data for GRR analysis with simplified network structure.

This script:
1. Loads village-level data and geographic data
2. Builds the adjacency graph G based on:
   - Nearest k villages within distance threshold (symmetric union rule)
   - Degree capped at max_degree (keeping closest neighbors)
   - No saturation consideration in network building
3. Computes exposure mapping (A, H) for each village:
   - A: self treatment
   - H: whether more than 1/3 of connected villages are treated
4. Collects covariates and outcomes
5. Saves processed data for GRR analysis

Output files:
- village_data.csv: Village-level data with covariates, outcomes, and exposure mapping
- adjacency_matrix.npy: Dense adjacency matrix G
- processing_info.json: Metadata

Usage:
    python preprocess.py

Note: Requires data/cash-transfer/analysisdata/ folder with:
    - GE_VillageLevel_ECMA.dta
    - village_buffers_hhs_ge.dta
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import sys
import json
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DATA_DIR = os.path.join(REPO_DIR, "data", "cash-transfer", "analysisdata")
OUTPUT_DIR = os.path.join(REPO_DIR, "data", "cash-transfer-derived", "processed_data")

# Network construction parameters
DISTANCE_THRESHOLD = 8  # km (strictly less than)
K_NEIGHBORS = 5  # number of nearest neighbors within distance threshold
MAX_DEGREE = 7  # cap degree at this value (keep closest neighbors)

# Outcomes to analyze
OUTCOMES_DICT = {
    'profit_per_ent': ('vill_profit1_wins', 'Profit per enterprise (KES)'),
    'revenue_per_ent': ('vill_revenue1_wins', 'Revenue per enterprise (KES)'),
    'cost_per_ent': ('vill_totcost_wins', 'Total costs per enterprise (KES)'),
    'wagebill_per_ent': ('vill_wagebill_wins', 'Wage bill per enterprise (KES)'),
}

# Baseline covariates
BASELINE_COVARIATES = [
    'n_allents_BL', 
    'vill_profit1_wins_BL', 
    'vill_revenue1_wins_BL', 
    'vill_totcost_wins_BL'
]


# =============================================================================
# Main Processing Functions
# =============================================================================

def load_data():
    """Load village-level and geographic data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    village_file = os.path.join(DATA_DIR, "GE_VillageLevel_ECMA.dta")
    geo_file = os.path.join(DATA_DIR, "village_buffers_hhs_ge.dta")
    
    if not os.path.exists(village_file):
        raise FileNotFoundError(
            f"Village data file not found: {village_file}\n"
            "Please ensure the data/cash-transfer/analysisdata/ folder exists "
            "with the required Stata files."
        )
    
    if not os.path.exists(geo_file):
        raise FileNotFoundError(
            f"Geographic data file not found: {geo_file}\n"
            "Please ensure the data/cash-transfer/analysisdata/ folder exists "
            "with the required Stata files."
        )
    
    print(f"Loading village data from: {village_file}")
    VillageLevel = pd.read_stata(village_file)
    print(f"  Loaded {len(VillageLevel)} rows")
    
    print(f"Loading geographic data from: {geo_file}")
    GEO = pd.read_stata(geo_file)
    print(f"  Loaded {len(GEO)} rows")
    
    return VillageLevel, GEO


def build_village_info(VillageLevel):
    """Extract unique villages and their basic info."""
    print("\n" + "=" * 80)
    print("BUILDING VILLAGE INFO")
    print("=" * 80)
    
    # Get unique villages
    villages = VillageLevel['village_code'].unique()
    n_villages = len(villages)
    print(f"Number of unique villages: {n_villages}")
    
    # Create village info dataframe
    village_info = VillageLevel[['village_code', 'sublocation_code', 'hi_sat', 'treat']].drop_duplicates()
    village_info = village_info.set_index('village_code')
    
    # Create mapping from village_code to index
    village_to_idx = {village: idx for idx, village in enumerate(villages)}
    idx_to_village = {idx: village for village, idx in village_to_idx.items()}
    
    # Create sublocation -> villages mapping (for reference, not used in network)
    sublocation_villages = {}
    for village in villages:
        subloc = village_info.loc[village, 'sublocation_code']
        if subloc not in sublocation_villages:
            sublocation_villages[subloc] = []
        sublocation_villages[subloc].append(village)
    
    print(f"Number of sublocations: {len(sublocation_villages)}")
    subloc_sizes = [len(v) for v in sublocation_villages.values()]
    print(f"Villages per sublocation: min={min(subloc_sizes)}, max={max(subloc_sizes)}, "
          f"mean={np.mean(subloc_sizes):.1f}")
    
    return villages, village_info, village_to_idx, idx_to_village, sublocation_villages


def find_nearest_villages(villages, village_info, GEO, village_to_idx, distance_threshold, k_neighbors):
    """
    Find nearest k villages within distance threshold for each village.
    
    Network construction rule:
    - For each village, find its nearest k villages within distance_threshold km
    - Add edges symmetrically: if A is in B's nearest k OR B is in A's nearest k, add edge (A,B)
    - No saturation/sublocation consideration
    
    Returns:
        nearest_villages: dict mapping village -> list of nearest villages
        village_distances: dict mapping (village1, village2) -> distance
    """
    print(f"\nFinding nearest {k_neighbors} villages within {distance_threshold}km...")
    
    # First, compute nearest k for each village (one-sided)
    nearest_villages = {}
    village_distances = {}  # Store distances for degree capping later
    
    for village in villages:
        # Get all villages in GEO data for this village
        village_geo = GEO[GEO['treatvill_code'] == village].copy()
        
        # Filter to only other villages (not self)
        village_geo = village_geo[village_geo['village_code'] != village]
        
        # Filter to only villages in our dataset
        village_geo = village_geo[village_geo['village_code'].isin(villages)]
        
        # Filter to only villages with distance < threshold
        village_geo = village_geo[village_geo['distance'] < distance_threshold]
        
        # Sort by distance and take at most k closest
        village_geo = village_geo.sort_values('distance')
        nearest = village_geo['village_code'].head(k_neighbors).tolist()
        
        nearest_villages[village] = nearest
        
        # Store all distances within threshold for degree capping
        for _, row in village_geo.iterrows():
            other_village = row['village_code']
            dist = row['distance']
            # Store symmetrically with sorted tuple key
            key = tuple(sorted([village, other_village]))
            if key not in village_distances:
                village_distances[key] = dist
    
    # Print one-sided statistics
    nearest_counts = [len(v) for v in nearest_villages.values()]
    print(f"  One-sided nearest neighbors:")
    print(f"    Average: {np.mean(nearest_counts):.2f}")
    print(f"    Min: {min(nearest_counts)}, Max: {max(nearest_counts)}")
    print(f"    Villages with {k_neighbors} neighbors: {sum([1 for c in nearest_counts if c == k_neighbors])}/{len(nearest_counts)}")
    print(f"    Villages with 0 neighbors: {sum([1 for c in nearest_counts if c == 0])}/{len(nearest_counts)}")
    
    return nearest_villages, village_distances


def build_adjacency_matrix(villages, village_to_idx, nearest_villages, village_distances, 
                           idx_to_village, max_degree=None):
    """
    Build symmetric adjacency matrix from nearest neighbor lists with optional degree capping.
    
    Uses union rule: edge (i, j) exists if i is in j's nearest k OR j is in i's nearest k
    
    If max_degree is specified, for villages exceeding this degree, we keep only the 
    closest max_degree neighbors (by distance).
    """
    print("\n" + "=" * 80)
    print("BUILDING ADJACENCY MATRIX")
    print("=" * 80)
    print(f"  Edge rule: union (either nominates)")
    if max_degree:
        print(f"  Degree cap: {max_degree} (keep closest neighbors)")
    
    n = len(villages)
    
    # Build adjacency matrix (with self-loops on diagonal)
    G = np.eye(n, dtype=int)
    
    # Union rule: edge if either nominates
    for village, near_list in nearest_villages.items():
        idx1 = village_to_idx[village]
        for near_village in near_list:
            if near_village in village_to_idx:
                idx2 = village_to_idx[near_village]
                G[idx1, idx2] = 1
                G[idx2, idx1] = 1
    
    # Print pre-cap statistics
    degrees_pre = G.sum(axis=1) - 1
    print(f"\n  Before capping:")
    print(f"    Edges: {int((G.sum() - n) // 2)}, Avg degree: {degrees_pre.mean():.2f}, Max: {int(degrees_pre.max())}")
    
    # Apply degree cap if specified
    if max_degree is not None:
        n_capped = 0
        for i in range(n):
            degree = int(G[i, :].sum() - 1)  # Exclude self-loop
            if degree > max_degree:
                n_capped += 1
                village = idx_to_village[i]
                
                # Find all current neighbors and their distances
                neighbors = []
                for j in range(n):
                    if i != j and G[i, j] == 1:
                        other_village = idx_to_village[j]
                        key = tuple(sorted([village, other_village]))
                        dist = village_distances.get(key, float('inf'))
                        neighbors.append((j, dist))
                
                # Sort by distance and keep only closest max_degree
                neighbors.sort(key=lambda x: x[1])
                keep_indices = set([nb[0] for nb in neighbors[:max_degree]])
                
                # Remove edges to neighbors beyond max_degree
                for j in range(n):
                    if i != j and G[i, j] == 1 and j not in keep_indices:
                        G[i, j] = 0
                        G[j, i] = 0  # Keep symmetric
        
        print(f"\n  After capping (capped {n_capped} villages):")
    
    # Compute statistics
    n_edges = (G.sum() - n) // 2  # Exclude diagonal
    degrees = G.sum(axis=1) - 1  # Exclude self-loop
    avg_degree = degrees.mean()
    max_degree = degrees.max()
    min_degree = degrees.min()
    
    print(f"Adjacency matrix statistics:")
    print(f"  Number of edges: {n_edges}")
    print(f"  Average degree: {avg_degree:.2f}")
    print(f"  Min degree: {min_degree}, Max degree: {max_degree}")
    print(f"  Degree distribution:")
    for d in range(int(max_degree) + 1):
        count = (degrees == d).sum()
        if count > 0:
            print(f"    Degree {d}: {count} villages ({count/n*100:.1f}%)")
    
    return G


def compute_exposure_mapping(VillageLevel, villages, village_info, village_to_idx, G):
    """
    Compute exposure mapping (A, H) for each village.
    
    A: self treatment (from data)
    H: 1 if more than 1/3 of connected villages are treated
    
    D = A * 2 + H gives values 0-3 for (A, H) combinations:
    - D=0: (A=0, H=0)
    - D=1: (A=0, H=1)
    - D=2: (A=1, H=0)
    - D=3: (A=1, H=1)
    """
    print("\n" + "=" * 80)
    print("COMPUTING EXPOSURE MAPPING (A, H) - H = more than 1/3 neighbors treated")
    print("=" * 80)
    
    n = len(villages)
    
    # Get treatment assignment A from data
    A = np.zeros(n, dtype=int)
    for village in villages:
        idx = village_to_idx[village]
        A[idx] = int(village_info.loc[village, 'treat'])
    
    print(f"Treatment: {A.sum()} treated, {n - A.sum()} control")
    
    # Compute H: neighborhood spillover
    # H_i = 1 if more than 1/3 of connected villages (excluding self) are treated
    H = np.zeros(n, dtype=int)
    
    for i in range(n):
        # Get neighbors (excluding self)
        neighbors = np.where(G[i, :] == 1)[0]
        neighbors = neighbors[neighbors != i]  # Exclude self
        
        if len(neighbors) > 0:
            treated_count = A[neighbors].sum()
            H[i] = 1 if treated_count > len(neighbors) / 3 else 0  # More than 1/3 treated
    
    print(f"Neighborhood spillover H: {H.sum()} with H=1 (more than 1/3 neighbors treated)")
    
    # Compute exposure D as combined indicator
    # D = A * 2 + H (gives values 0-3 for (A, H) combinations)
    D = A * 2 + H
    
    # Print distribution
    print("\nExposure distribution (A, H):")
    for d in range(4):
        a = d // 2
        h = d % 2
        count = (D == d).sum()
        print(f"  (A={a}, H={h}): {count} villages")
    
    return A, H, D


def extract_outcomes_and_covariates(VillageLevel, villages, village_to_idx):
    """Extract outcomes and baseline covariates."""
    print("\n" + "=" * 80)
    print("EXTRACTING OUTCOMES AND COVARIATES")
    print("=" * 80)
    
    n = len(villages)
    
    # Get unique village data
    outcome_cols = ['village_code', 'n_allents'] + [v[0] for v in OUTCOMES_DICT.values()]
    outcomes_data = VillageLevel[outcome_cols].drop_duplicates(subset='village_code')
    
    # Create per-enterprise versions
    for key, (var, label) in OUTCOMES_DICT.items():
        outcomes_data[key] = outcomes_data[var] / outcomes_data['n_allents']
    
    # Get baseline covariates
    covariate_data = VillageLevel[['village_code'] + BASELINE_COVARIATES].drop_duplicates(subset='village_code')
    
    # Create per-enterprise baseline versions
    covariate_data['profit_BL_per_ent'] = covariate_data['vill_profit1_wins_BL'] / covariate_data['n_allents_BL']
    covariate_data['revenue_BL_per_ent'] = covariate_data['vill_revenue1_wins_BL'] / covariate_data['n_allents_BL']
    covariate_data['cost_BL_per_ent'] = covariate_data['vill_totcost_wins_BL'] / covariate_data['n_allents_BL']
    
    # Merge
    merged = outcomes_data.merge(covariate_data, on='village_code')
    
    print(f"Extracted {len(merged)} villages with outcomes and covariates")
    print(f"Outcomes: {list(OUTCOMES_DICT.keys())}")
    print(f"Baseline covariates: {BASELINE_COVARIATES}")
    
    return merged


def save_processed_data(villages, village_to_idx, G, A, H, D, outcomes_covariates, village_info, output_dir):
    """Save all processed data."""
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save village data (exposure mapping + outcomes + covariates)
    village_data = pd.DataFrame({
        'village_code': villages,
        'village_idx': [village_to_idx[v] for v in villages],
        'sublocation_code': [village_info.loc[v, 'sublocation_code'] for v in villages],
        'hi_sat': [int(village_info.loc[v, 'hi_sat']) for v in villages],
        'A': A,
        'H': H,
        'D': D,
    })
    
    # Merge with outcomes and covariates
    village_data = village_data.merge(outcomes_covariates, on='village_code')
    
    village_data_path = os.path.join(output_dir, "village_data.csv")
    village_data.to_csv(village_data_path, index=False)
    print(f"Saved village data to: {village_data_path}")
    
    # 2. Save adjacency matrix
    adj_path = os.path.join(output_dir, "adjacency_matrix.npy")
    np.save(adj_path, G)
    print(f"Saved adjacency matrix to: {adj_path}")
    
    # Also save as sparse
    sp.save_npz(os.path.join(output_dir, "adjacency_matrix.npz"), sp.csr_matrix(G))
    
    # 3. Save metadata
    n = len(villages)
    degrees = G.sum(axis=1) - 1
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_villages': n,
        'n_sublocations': len(set(village_info['sublocation_code'])),
        'network_construction': {
            'distance_threshold_km': DISTANCE_THRESHOLD,
            'k_neighbors': K_NEIGHBORS,
            'rule': 'Nearest k within distance threshold, symmetric edges',
        },
        'exposure_mapping': {
            'description': '(A, H) where A is self-treatment, H is more than 1/3 neighbors treated',
            'n_levels': 4,
            'encoding': 'D = A * 2 + H',
        },
        'graph_statistics': {
            'n_edges': int((G.sum() - n) // 2),
            'avg_degree': float(degrees.mean()),
            'max_degree': int(degrees.max()),
            'min_degree': int(degrees.min()),
        },
        'exposure_distribution': {
            f"(A={d//2},H={d%2})": int((D == d).sum()) for d in range(4)
        },
        'outcomes': list(OUTCOMES_DICT.keys()),
        'baseline_covariates': BASELINE_COVARIATES,
    }
    
    metadata_path = os.path.join(output_dir, "processing_info.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    return village_data


# =============================================================================
# Main
# =============================================================================

def main():
    """Main preprocessing pipeline."""
    print("\n" + "=" * 80)
    print("INTERFERENCE-CASH TRANSFER DATA PREPROCESSING")
    print(f"SIMPLIFIED NETWORK: Nearest {K_NEIGHBORS} within {DISTANCE_THRESHOLD}km, max degree {MAX_DEGREE}")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Distance threshold: {DISTANCE_THRESHOLD} km")
    print(f"Max neighbors: {K_NEIGHBORS}")
    
    try:
        # Load data
        VillageLevel, GEO = load_data()
        
        # Build village info
        villages, village_info, village_to_idx, idx_to_village, sublocation_villages = build_village_info(VillageLevel)
        
        # Find nearest villages
        nearest_villages, village_distances = find_nearest_villages(
            villages, village_info, GEO, village_to_idx, 
            DISTANCE_THRESHOLD, K_NEIGHBORS
        )
        
        # Build adjacency matrix with degree capping
        G = build_adjacency_matrix(
            villages, village_to_idx, nearest_villages, village_distances,
            idx_to_village, max_degree=MAX_DEGREE
        )
        
        # Compute exposure mapping
        A, H, D = compute_exposure_mapping(VillageLevel, villages, village_info, village_to_idx, G)
        
        # Extract outcomes and covariates
        outcomes_covariates = extract_outcomes_and_covariates(VillageLevel, villages, village_to_idx)
        
        # Save processed data
        village_data = save_processed_data(
            villages, village_to_idx, G, A, H, D, 
            outcomes_covariates, village_info, OUTPUT_DIR
        )
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Output directory: {OUTPUT_DIR}")
        print("\nSaved files:")
        print("  - village_data.csv: Village data with (A, H) exposure and outcomes")
        print("  - adjacency_matrix.npy: Adjacency matrix G")
        print("  - processing_info.json: Metadata")
        print("\nNext steps:")
        print("  1. Run compute_inclusion_prob.py to generate propensity scores")
        print("  2. Run GRR-analysis.py to estimate treatment effects")
        
        return village_data, G
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo obtain the data files:")
        print("  1. Download from the original study's replication materials")
        print("  2. Place in: <repo>/data/cash-transfer/analysisdata/")
        print("  3. Required files:")
        print("     - GE_VillageLevel_ECMA.dta")
        print("     - village_buffers_hhs_ge.dta")
        sys.exit(1)


if __name__ == "__main__":
    main()
