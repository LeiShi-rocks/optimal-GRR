import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
VillageLevel = pd.read_stata("/Users/leis/Documents/GitHub/Interference-CashTransfer/replication_materials/analysisdata/GE_VillageLevel_ECMA.dta")
GEO = pd.read_stata("/Users/leis/Documents/GitHub/Interference-CashTransfer/replication_materials/analysisdata/village_buffers_hhs_ge.dta")

print(f"\nGEO columns: {GEO.columns.tolist()}")
print(f"\nGEO shape: {GEO.shape}")

# Create village info
village_info = VillageLevel[['village_code', 'sublocation_code', 'hi_sat']].drop_duplicates()
village_info = village_info.set_index('village_code')
villages = VillageLevel['village_code'].unique()

print(f"\nNumber of unique villages: {len(villages)}")

# Check distance statistics
print("\n" + "="*80)
print("DISTANCE STATISTICS IN GEO DATA")
print("="*80)
print(f"\nDistance column statistics:")
print(GEO['distance'].describe())

print(f"\nDistance range: [{GEO['distance'].min():.4f}, {GEO['distance'].max():.4f}]")

# Check how many villages are within different distance thresholds
thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
print("\n" + "="*80)
print("NEIGHBORS AT DIFFERENT DISTANCE THRESHOLDS")
print("="*80)

for threshold in thresholds:
    neighbor_counts = []
    
    for village in villages:
        subloc = village_info.loc[village, 'sublocation_code']
        
        # Get villages for this village
        village_geo = GEO[GEO['treatvill_code'] == village].copy()
        
        # Filter out self and same sublocation
        village_geo = village_geo[village_geo['village_code'] != village]
        village_geo['target_sublocation'] = village_geo['village_code'].map(
            lambda x: village_info.loc[x, 'sublocation_code'] if x in village_info.index else None
        )
        village_geo = village_geo[village_geo['target_sublocation'] != subloc]
        
        # Count within threshold
        within_threshold = village_geo[village_geo['distance'] <= threshold]
        neighbor_counts.append(len(within_threshold))
    
    print(f"\nThreshold <= {threshold:2d}:")
    print(f"  Avg neighbors: {np.mean(neighbor_counts):.2f}")
    print(f"  Min: {np.min(neighbor_counts)}, Max: {np.max(neighbor_counts)}")
    print(f"  Villages with 0 neighbors: {sum([1 for c in neighbor_counts if c == 0])}")
    print(f"  Villages with >= 5 neighbors: {sum([1 for c in neighbor_counts if c >= 5])}")

# Let's look at a specific village example
print("\n" + "="*80)
print("EXAMPLE VILLAGE ANALYSIS")
print("="*80)

sample_village = villages[0]
print(f"\nAnalyzing village: {sample_village}")

village_geo = GEO[GEO['treatvill_code'] == sample_village].copy()
subloc = village_info.loc[sample_village, 'sublocation_code']

# Filter appropriately
village_geo = village_geo[village_geo['village_code'] != sample_village]
village_geo['target_sublocation'] = village_geo['village_code'].map(
    lambda x: village_info.loc[x, 'sublocation_code'] if x in village_info.index else None
)

print(f"  Sublocation: {subloc}")
print(f"  Total other villages in data: {len(village_geo)}")

same_subloc = village_geo[village_geo['target_sublocation'] == subloc]
diff_subloc = village_geo[village_geo['target_sublocation'] != subloc]

print(f"  Villages in same sublocation: {len(same_subloc)}")
print(f"  Villages in different sublocations: {len(diff_subloc)}")

if len(diff_subloc) > 0:
    print(f"\n  Distance to different-sublocation villages:")
    distances_sorted = diff_subloc.sort_values('distance')
    print(f"    Nearest 10 distances: {distances_sorted['distance'].head(10).tolist()}")
    print(f"    Distances <= 6: {len(diff_subloc[diff_subloc['distance'] <= 6])}")
    print(f"    Distances <= 3: {len(diff_subloc[diff_subloc['distance'] <= 3])}")

# Check if distance might be in different units
print("\n" + "="*80)
print("CHECKING DISTANCE UNITS")
print("="*80)
print("\nNote: If distance is in kilometers, then 6 km is quite a reasonable threshold.")
print("If distance is in some other unit, the interpretation changes.")
print("\nLet's check the distribution of distances for different-sublocation pairs:")

all_diff_subloc_distances = []
for village in villages[:50]:  # Sample first 50 villages for speed
    subloc = village_info.loc[village, 'sublocation_code']
    village_geo = GEO[GEO['treatvill_code'] == village].copy()
    village_geo = village_geo[village_geo['village_code'] != village]
    village_geo['target_sublocation'] = village_geo['village_code'].map(
        lambda x: village_info.loc[x, 'sublocation_code'] if x in village_info.index else None
    )
    diff_subloc = village_geo[village_geo['target_sublocation'] != subloc]
    all_diff_subloc_distances.extend(diff_subloc['distance'].tolist())

if all_diff_subloc_distances:
    print(f"\nDistances to different-sublocation neighbors (sample):")
    print(f"  Count: {len(all_diff_subloc_distances)}")
    print(f"  Mean: {np.mean(all_diff_subloc_distances):.4f}")
    print(f"  Median: {np.median(all_diff_subloc_distances):.4f}")
    print(f"  Std: {np.std(all_diff_subloc_distances):.4f}")
    print(f"  Min: {np.min(all_diff_subloc_distances):.4f}")
    print(f"  Max: {np.max(all_diff_subloc_distances):.4f}")
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(all_diff_subloc_distances, p)
        print(f"    {p}th: {val:.4f}")


