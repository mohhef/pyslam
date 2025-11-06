#!/bin/bash

# Script to test different feature detectors on clean/rain/fog datasets
# Usage: ./test_features.sh

DETECTORS=("ORB" "SIFT" "ROOT_SIFT" "AKAZE" "BRISK")
DATASETS=("clean" "rain" "fog")

# Base paths
CLEAN_PATH="/home/mohhef/projects/SLAMWeatherBench/datasets/kitti"
RAIN_PATH="/home/mohhef/projects/SLAMWeatherBench/results/kitti_rain_physics_full"
FOG_PATH="/home/mohhef/projects/SLAMWeatherBench/results/kitti_static_fog"

CONFIG_FILE="/home/mohhef/projects/pyslam/settings/KITTI04-12.yaml"
RESULTS_BASE="/home/mohhef/projects/pyslam/results"

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
            sed -i "s|base_path:.*kitti.*|base_path: $CLEAN_PATH|" /home/mohhef/projects/pyslam/config.yaml
        elif [ "$dataset" == "rain" ]; then
            sed -i "s|base_path:.*kitti.*|base_path: $RAIN_PATH|" /home/mohhef/projects/pyslam/config.yaml
        elif [ "$dataset" == "fog" ]; then
            sed -i "s|base_path:.*kitti.*|base_path: $FOG_PATH|" /home/mohhef/projects/pyslam/config.yaml
        fi

        # Run VO
        cd /home/mohhef/projects/pyslam
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
