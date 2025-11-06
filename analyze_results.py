#!/usr/bin/env python3
"""
Analyze and compare feature tracking results across different detectors and conditions.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

results_dir = "results"

detectors = ["ORB", "SIFT", "ROOT_SIFT", "AKAZE", "BRISK"]
datasets = ["clean", "rain", "fog"]

# Collect statistics
results = {}

for detector in detectors:
    results[detector] = {}
    for dataset in datasets:
        json_file = f"{results_dir}/tracks_{detector}_{dataset}.json"

        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                stats = data.get('statistics', {})
                results[detector][dataset] = {
                    'mean_track_length': stats.get('mean_track_length', 0),
                    'median_track_length': stats.get('median_track_length', 0),
                    'max_track_length': stats.get('max_track_length', 0),
                    'total_tracks': stats.get('total_tracks', 0)
                }
        else:
            print(f"Warning: Missing {json_file}")
            results[detector][dataset] = None

# Print comparison table
print("\n" + "="*80)
print("FEATURE TRACKING COMPARISON: Mean Track Length (frames)")
print("="*80)
print(f"{'Detector':<15} {'Clean':<10} {'Rain':<10} {'Fog':<10} {'Rain Degradation':<20} {'Fog Degradation':<20}")
print("-"*80)

for detector in detectors:
    if all(results[detector].get(ds) for ds in datasets):
        clean_mean = results[detector]['clean']['mean_track_length']
        rain_mean = results[detector]['rain']['mean_track_length']
        fog_mean = results[detector]['fog']['mean_track_length']

        rain_deg = ((clean_mean - rain_mean) / clean_mean * 100) if clean_mean > 0 else 0
        fog_deg = ((clean_mean - fog_mean) / clean_mean * 100) if clean_mean > 0 else 0

        print(f"{detector:<15} {clean_mean:<10.2f} {rain_mean:<10.2f} {fog_mean:<10.2f} {rain_deg:>6.1f}% {'':<13} {fog_deg:>6.1f}%")
    else:
        print(f"{detector:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

print("="*80)

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Mean Track Length
ax = axes[0]
x = np.arange(len(detectors))
width = 0.25

clean_means = [results[d]['clean']['mean_track_length'] if results[d].get('clean') else 0 for d in detectors]
rain_means = [results[d]['rain']['mean_track_length'] if results[d].get('rain') else 0 for d in detectors]
fog_means = [results[d]['fog']['mean_track_length'] if results[d].get('fog') else 0 for d in detectors]

ax.bar(x - width, clean_means, width, label='Clean', color='green', alpha=0.8)
ax.bar(x, rain_means, width, label='Rain', color='blue', alpha=0.8)
ax.bar(x + width, fog_means, width, label='Fog', color='gray', alpha=0.8)

ax.set_xlabel('Feature Detector', fontsize=12)
ax.set_ylabel('Mean Track Length (frames)', fontsize=12)
ax.set_title('Feature Tracking Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(detectors, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Degradation Percentage
ax = axes[1]
rain_degradation = []
fog_degradation = []

for detector in detectors:
    if all(results[detector].get(ds) for ds in datasets):
        clean = results[detector]['clean']['mean_track_length']
        rain = results[detector]['rain']['mean_track_length']
        fog = results[detector]['fog']['mean_track_length']

        rain_deg = ((clean - rain) / clean * 100) if clean > 0 else 0
        fog_deg = ((clean - fog) / clean * 100) if clean > 0 else 0

        rain_degradation.append(rain_deg)
        fog_degradation.append(fog_deg)
    else:
        rain_degradation.append(0)
        fog_degradation.append(0)

ax.bar(x - width/2, rain_degradation, width, label='Rain', color='blue', alpha=0.8)
ax.bar(x + width/2, fog_degradation, width, label='Fog', color='gray', alpha=0.8)

ax.set_xlabel('Feature Detector', fontsize=12)
ax.set_ylabel('Performance Degradation (%)', fontsize=12)
ax.set_title('Robustness to Weather Conditions', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(detectors, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Plot 3: Max Track Length
ax = axes[2]
clean_max = [results[d]['clean']['max_track_length'] if results[d].get('clean') else 0 for d in detectors]
rain_max = [results[d]['rain']['max_track_length'] if results[d].get('rain') else 0 for d in detectors]
fog_max = [results[d]['fog']['max_track_length'] if results[d].get('fog') else 0 for d in detectors]

ax.bar(x - width, clean_max, width, label='Clean', color='green', alpha=0.8)
ax.bar(x, rain_max, width, label='Rain', color='blue', alpha=0.8)
ax.bar(x + width, fog_max, width, label='Fog', color='gray', alpha=0.8)

ax.set_xlabel('Feature Detector', fontsize=12)
ax.set_ylabel('Max Track Length (frames)', fontsize=12)
ax.set_title('Longest Tracked Feature', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(detectors, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{results_dir}/comparison_all_detectors.png', dpi=150, bbox_inches='tight')
print(f"\nComparison plot saved to: {results_dir}/comparison_all_detectors.png")

# Find best performer
print("\n" + "="*80)
print("BEST PERFORMERS")
print("="*80)

best_clean = max(detectors, key=lambda d: results[d]['clean']['mean_track_length'] if results[d].get('clean') else 0)
best_rain = max(detectors, key=lambda d: results[d]['rain']['mean_track_length'] if results[d].get('rain') else 0)
best_fog = max(detectors, key=lambda d: results[d]['fog']['mean_track_length'] if results[d].get('fog') else 0)

print(f"Best in Clean conditions: {best_clean}")
print(f"Best in Rain conditions:  {best_rain}")
print(f"Best in Fog conditions:   {best_fog}")

# Most robust (least degradation)
min_degradation = {}
for detector in detectors:
    if all(results[detector].get(ds) for ds in datasets):
        clean = results[detector]['clean']['mean_track_length']
        rain = results[detector]['rain']['mean_track_length']
        fog = results[detector]['fog']['mean_track_length']
        avg_degradation = (((clean - rain) / clean) + ((clean - fog) / clean)) / 2 * 100 if clean > 0 else 100
        min_degradation[detector] = avg_degradation

most_robust = min(min_degradation.keys(), key=lambda d: min_degradation[d])
print(f"Most robust (least degradation): {most_robust} ({min_degradation[most_robust]:.1f}% avg degradation)")
print("="*80 + "\n")
