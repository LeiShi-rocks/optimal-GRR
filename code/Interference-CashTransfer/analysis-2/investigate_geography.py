import pandas as pd
import numpy as np

# Load data
print("Loading data...")
VillageLevel = pd.read_stata("/Users/leis/Documents/GitHub/Interference-CashTransfer/replication_materials/analysisdata/GE_VillageLevel_ECMA.dta")
GEO = pd.read_stata("/Users/leis/Documents/GitHub/Interference-CashTransfer/replication_materials/analysisdata/village_buffers_hhs_ge.dta")

print(f"\nVillageLevel shape: {VillageLevel.shape}")
print(f"GEO shape: {GEO.shape}")
print(f"Unique villages: {len(VillageLevel['village_code'].unique())}")
print(f"Unique sublocations: {len(VillageLevel['sublocation_code'].unique())}")

# Check the structure of GEO
print("\n" + "="*80)
print("GEO DATA STRUCTURE")
print("="*80)
print(f"\nUnique treatvill_code values: {len(GEO['treatvill_code'].unique())}")
print(f"Unique village_code values in GEO: {len(GEO['village_code'].unique())}")

# For one village, how many rows?
sample_village = GEO['treatvill_code'].iloc[0]
sample_rows = GEO[GEO['treatvill_code'] == sample_village]
print(f"\nFor village {sample_village}:")
print(f"  Number of rows: {len(sample_rows)}")
print(f"  This includes the village itself? {sample_village in sample_rows['village_code'].values}")

# Check what distance bins actually mean
print("\n" + "="*80)
print("UNDERSTANDING DISTANCE BINS")
print("="*80)
print("\nDistance values are binned. Let's see what each bin contains:")

village_info = VillageLevel[['village_code', 'sublocation_code']].drop_duplicates()
village_info = village_info.set_index('village_code')

# Pick one village and look at distances
sample_village = VillageLevel['village_code'].unique()[0]
village_geo = GEO[GEO['treatvill_code'] == sample_village].copy()

# Add sublocation info
subloc = village_info.loc[sample_village, 'sublocation_code']
village_geo['target_sublocation'] = village_geo['village_code'].map(
    lambda x: village_info.loc[x, 'sublocation_code'] if x in village_info.index else None
)

# Separate same vs different sublocation
same_subloc = village_geo[village_geo['target_sublocation'] == subloc]
diff_subloc = village_geo[village_geo['target_sublocation'] != subloc]

print(f"\nExample village {sample_village} (sublocation {subloc}):")
print(f"  Total other villages in GEO: {len(village_geo) - 1}")  # -1 for self
print(f"  Same sublocation: {len(same_subloc) - 1}")  # -1 if self is included
print(f"  Different sublocation: {len(diff_subloc)}")

print("\nDistance distribution for DIFFERENT sublocation villages:")
for dist in sorted(diff_subloc['distance'].unique()):
    count = len(diff_subloc[diff_subloc['distance'] == dist])
    print(f"  Distance = {dist}: {count} villages")

# Check geographic spread
print("\n" + "="*80)
print("GEOGRAPHIC INTERPRETATION")
print("="*80)

# Calculate how many villages per square km this implies
print("\nIf distances are binned (e.g., '6' means villages in the 4-6 km band):")
print("  A 6km radius circle has area = π * 6² ≈ 113 square km")
print("  A 4km radius circle has area = π * 4² ≈ 50 square km")
print("  So the '6 km band' might mean villages in the annulus of 50-113 sq km")
print("  That's about 63 square km of area")

# What's the density?
avg_neighbors_dist6 = diff_subloc[diff_subloc['distance'] <= 6].shape[0]
print(f"\nFor our example village:")
print(f"  Villages within distance ≤ 6 (different sublocation): {avg_neighbors_dist6}")
print(f"  If 6km radius circle (113 sq km), density ≈ {avg_neighbors_dist6/113:.1f} villages per sq km")

# Check if this dataset might be showing ALL pairwise distances, not just nearby
print("\n" + "="*80)
print("HYPOTHESIS: Is GEO showing ALL villages in the study area?")
print("="*80)

print(f"\nTotal villages in study: {len(VillageLevel['village_code'].unique())}")
print(f"Average rows per treatvill_code in GEO: {len(GEO) / len(GEO['treatvill_code'].unique()):.1f}")
print("\nThis suggests GEO contains distance information for MOST (not just nearby) villages")
print("The 'distance' field might be binned/rounded distances to ALL villages in the study area.")

# What percentage of villages are within 6km?
villages = VillageLevel['village_code'].unique()
n_villages = len(villages)

within_6km_counts = []
for village in villages[:50]:  # Sample first 50
    subloc = village_info.loc[village, 'sublocation_code']
    village_geo = GEO[GEO['treatvill_code'] == village].copy()
    village_geo['target_sublocation'] = village_geo['village_code'].map(
        lambda x: village_info.loc[x, 'sublocation_code'] if x in village_info.index else None
    )
    diff_subloc = village_geo[village_geo['target_sublocation'] != subloc]
    within_6km_counts.append(len(diff_subloc[diff_subloc['distance'] <= 6]))

print(f"\nSample of 50 villages:")
print(f"  Average villages within 6km (different sublocation): {np.mean(within_6km_counts):.1f}")
print(f"  This is {np.mean(within_6km_counts)/n_villages*100:.1f}% of all villages")

# Check the max distance
print(f"\n" + "="*80)
print(f"MAX DISTANCE IN DATA")
print(f"="*80)
print(f"Maximum distance value: {GEO['distance'].max()} km")
print(f"This suggests the study area spans roughly {GEO['distance'].max()} km in diameter")
print(f"(or at least, the furthest villages are about {GEO['distance'].max()} km apart)")


