#!/bin/bash

# Script to test different feature detectors on clean/rain/fog datasets
# Usage: ./test_features.sh

DETECTORS=("ORB" "SIFT" "ROOT_SIFT" "AKAZE" "BRISK")
DATASETS=("clean" "rain" "fog")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYSLAM_ROOT="$SCRIPT_DIR"
SLAMLAB_ROOT="$(cd "$PYSLAM_ROOT/../../.." && pwd)"

# Base paths
CLEAN_PATH="$SLAMLAB_ROOT/datasets/kitti"
RAIN_PATH="$SLAMLAB_ROOT/results/kitti_rain_physics_full"
FOG_PATH="$SLAMLAB_ROOT/results/kitti_static_fog"

CONFIG_FILE="$PYSLAM_ROOT/settings/KITTI04-12.yaml"
RESULTS_BASE="$PYSLAM_ROOT/results"

for detector in "${DETECTORS[@]}"; do
    echo "========================================"
    echo "Testing detector: $detector"
    echo "========================================"

    for dataset in "${DATASETS[@]}"; do
        echo "  Dataset: $dataset"

        # Update config with current detector
        sed -i "s/FeatureTrackerConfig.name:.*/FeatureTrackerConfig.name: $detector/" "$CONFIG_FILE"

        # Update config with current dataset path
        if [ "$dataset" == "clean" ]; then
            sed -i "s|base_path:.*kitti.*|base_path: $CLEAN_PATH|" "$PYSLAM_ROOT/config.yaml"
        elif [ "$dataset" == "rain" ]; then
            sed -i "s|base_path:.*kitti.*|base_path: $RAIN_PATH|" "$PYSLAM_ROOT/config.yaml"
        elif [ "$dataset" == "fog" ]; then
            sed -i "s|base_path:.*kitti.*|base_path: $FOG_PATH|" "$PYSLAM_ROOT/config.yaml"
        fi

        # Run VO
        cd "$PYSLAM_ROOT"
        python main_vo.py

        # Rename output files to include detector and dataset
        if [ -f "$RESULTS_BASE/track_longevity_04.png" ]; then
            mv "$RESULTS_BASE/track_longevity_04.png" "$RESULTS_BASE/track_longevity_${detector}_${dataset}.png"
        fi

        if [ -f "$RESULTS_BASE/tracks_04.json" ]; then
            mv "$RESULTS_BASE/tracks_04.json" "$RESULTS_BASE/tracks_${detector}_${dataset}.json"
        fi

        echo "  Completed: $detector on $dataset"
        echo ""
    done
done

echo "========================================"
echo "All tests completed!"
echo "Results saved in: $RESULTS_BASE"
echo "========================================"
