#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2
import os
import math
import time
import platform
import json
import argparse

from pyslam.config import Config

from pyslam.slam.visual_odometry import VisualOdometryEducational
from pyslam.slam.visual_odometry_rgbd import (
    VisualOdometryRgbd,
    VisualOdometryRgbdTensor,
)
from pyslam.slam.camera import PinholeCamera

from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import DatasetType, SensorType

from pyslam.viz.mplot_thread import Mplot2d, Mplot3d
from pyslam.viz.qplot_thread import Qplot2d
from pyslam.viz.rerun_interface import Rerun

from pyslam.local_features.feature_tracker import (
    feature_tracker_factory,
    FeatureTrackerTypes,
)
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.utilities.utils_sys import Printer


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kResultsFolder = kRootFolder + "/results"  # Default, will be overridden by config if available


kUseRerun = False
# check rerun does not have issues
if kUseRerun and not Rerun.is_ok:
    kUseRerun = False

"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
kUsePangolin = True
if platform.system() == "Darwin":
    kUsePangolin = (
        True  # Under mac force pangolin to be used since Mplot3d() has some reliability issues
    )
if kUsePangolin:
    from pyslam.viz.viewer3D import Viewer3D

kUseQplot2d = False
if platform.system() == "Darwin":
    kUseQplot2d = True  # Under mac force the usage of Qtplot2d: It is smoother


def factory_plot2d(*args, **kwargs):
    if kUseRerun:
        return None
    if kUseQplot2d:
        return Qplot2d(*args, **kwargs)
    else:
        return Mplot2d(*args, **kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=None,
        help="Path to custom config.yaml file",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    args = parser.parse_args()

    if args.config_path:
        config = Config(args.config_path)
    else:
        config = Config()

    # Use SAVE_TRAJECTORY output_folder for plots if available
    if hasattr(config, 'trajectory_saving_settings') and config.trajectory_saving_settings:
        kResultsFolder = config.trajectory_saving_settings.get('output_folder', kResultsFolder)

    dataset = dataset_factory(config)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config)

    num_features = 2000  # how many features do you want to detect and track?
    if (
        config.num_features_to_extract > 0
    ):  # override the number of features to extract if we set something in the settings file
        num_features = config.num_features_to_extract

    # select your tracker configuration (see the file feature_tracker_configs.py)
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, LIGHTGLUE, XFEAT, XFEAT_XFEAT, LOFTR

    # Use tracker from config if specified, otherwise default to LK_SHI_TOMASI
    if config.feature_tracker_config_name:
        tracker_config = getattr(FeatureTrackerConfigs, config.feature_tracker_config_name)
    else:
        tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI

    tracker_config["num_features"] = num_features

    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object
    if dataset.sensor_type == SensorType.RGBD:
        vo = VisualOdometryRgbdTensor(cam, groundtruth)  # only for RGBD
        Printer.green("Using VisualOdometryRgbdTensor")
    else:
        vo = VisualOdometryEducational(cam, groundtruth, feature_tracker)
        Printer.green("Using VisualOdometryEducational")
    time.sleep(1)  # time to read the message

    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5 * traj_img_size)
    draw_scale = 1

    viewer3D = None
    plt3d = None
    err_plt = None
    matched_points_plt = None

    is_draw_3d = not args.headless
    is_draw_err = not args.headless
    is_draw_matched_points = not args.headless
    is_draw_with_rerun = kUseRerun and not args.headless

    if not args.headless:
        if is_draw_with_rerun:
            Rerun.init_vo()
        else:
            if kUsePangolin:
                viewer3D = Viewer3D(scale=dataset.scale_viewer_3d * 10)
            else:
                plt3d = Mplot3d(title="3D trajectory")

        err_plt = factory_plot2d(xlabel="img id", ylabel="m", title="error")
        matched_points_plt = factory_plot2d(xlabel="img id", ylabel="# matches", title="# matches")

    # Storage for post-processing track longevity visualization
    all_kps = []  # Store keypoints for each frame
    all_des = []  # Store descriptors for each frame
    all_images = []  # Store images for descriptor computation if needed

    img_id = 0
    while True:

        img = None

        if dataset.is_ok:
            timestamp = dataset.getTimestamp()  # get current timestamp
            img = dataset.getImageColor(img_id)
            depth = dataset.getDepth(img_id)
            img_right = (
                dataset.getImageColorRight(img_id)
                if dataset.sensor_type == SensorType.STEREO
                else None
            )

        if img is not None:

            vo.track(img, img_right, depth, img_id, timestamp)  # main VO function

            # Store data for post-processing track longevity analysis
            # Store the grayscale image
            if hasattr(vo, 'cur_image') and vo.cur_image is not None:
                all_images.append(vo.cur_image.copy())
            else:
                all_images.append(None)

            # Store keypoints and descriptors
            # On first frame: kps_ref/des_ref are set, kps_cur/des_cur are not
            # On subsequent frames: kps_cur/des_cur are set
            kps_to_store = None
            des_to_store = None

            if hasattr(vo, 'kps_cur') and vo.kps_cur is not None:
                kps_to_store = vo.kps_cur.copy()
                des_to_store = vo.des_cur.copy() if (hasattr(vo, 'des_cur') and vo.des_cur is not None) else None
            elif hasattr(vo, 'kps_ref') and vo.kps_ref is not None:
                # First frame - use ref instead of cur
                kps_to_store = vo.kps_ref.copy()
                des_to_store = vo.des_ref.copy() if (hasattr(vo, 'des_ref') and vo.des_ref is not None) else None

            all_kps.append(kps_to_store)
            all_des.append(des_to_store)

            if (
                len(vo.traj3d_est) > 1
            ):  # start drawing from the third image (when everything is initialized and flows in a normal way)

                x, y, z = vo.traj3d_est[-1]
                gt_x, gt_y, gt_z = vo.traj3d_gt[-1]

                if is_draw_traj_img:  # draw 2D trajectory (on the plane xz)
                    draw_x, draw_y = int(
                        draw_scale * x
                    ) + half_traj_img_size, half_traj_img_size - int(draw_scale * z)
                    draw_gt_x, draw_gt_y = int(
                        draw_scale * gt_x
                    ) + half_traj_img_size, half_traj_img_size - int(draw_scale * gt_z)
                    cv2.circle(
                        traj_img,
                        (draw_x, draw_y),
                        1,
                        (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0),
                        1,
                    )  # estimated from green to blue
                    cv2.circle(
                        traj_img, (draw_gt_x, draw_gt_y), 1, (0, 0, 255), 1
                    )  # groundtruth in red
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(
                        traj_img,
                        text,
                        (20, 40),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        1,
                        8,
                    )
                    # show

                    if is_draw_with_rerun:
                        Rerun.log_img_seq("trajectory_img/2d", img_id, traj_img)
                    elif not args.headless:
                        cv2.imshow("Trajectory", traj_img)

                if is_draw_with_rerun:
                    Rerun.log_2d_seq_scalar("trajectory_error/err_x", img_id, math.fabs(gt_x - x))
                    Rerun.log_2d_seq_scalar("trajectory_error/err_y", img_id, math.fabs(gt_y - y))
                    Rerun.log_2d_seq_scalar("trajectory_error/err_z", img_id, math.fabs(gt_z - z))

                    Rerun.log_2d_seq_scalar(
                        "trajectory_stats/num_matches", img_id, vo.num_matched_kps
                    )
                    Rerun.log_2d_seq_scalar("trajectory_stats/num_inliers", img_id, vo.num_inliers)

                    Rerun.log_3d_camera_img_seq(img_id, vo.draw_img, None, cam, vo.poses[-1])
                    Rerun.log_3d_trajectory(img_id, vo.traj3d_est, "estimated", color=[0, 0, 255])
                    Rerun.log_3d_trajectory(img_id, vo.traj3d_gt, "ground_truth", color=[255, 0, 0])
                else:
                    if is_draw_3d:  # draw 3d trajectory
                        if kUsePangolin:
                            viewer3D.draw_vo(vo)
                        else:
                            plt3d.draw(vo.traj3d_gt, "ground truth", color="r", marker=".")
                            plt3d.draw(vo.traj3d_est, "estimated", color="g", marker=".")

                    if is_draw_err:  # draw error signals
                        errx = [img_id, math.fabs(gt_x - x)]
                        erry = [img_id, math.fabs(gt_y - y)]
                        errz = [img_id, math.fabs(gt_z - z)]
                        err_plt.draw(errx, "err_x", color="g")
                        err_plt.draw(erry, "err_y", color="b")
                        err_plt.draw(errz, "err_z", color="r")

                    if is_draw_matched_points:
                        matched_kps_signal = [img_id, vo.num_matched_kps]
                        inliers_signal = [img_id, vo.num_inliers]
                        matched_points_plt.draw(matched_kps_signal, "# matches", color="b")
                        matched_points_plt.draw(inliers_signal, "# inliers", color="g")


            # draw camera image
            if not is_draw_with_rerun and not args.headless:
                cv2.imshow("Camera", vo.draw_img)

        else:
            time.sleep(0.1)
            if args.headless:
                break  # exit from the loop if headless and no more images

        if not args.headless:
            # get keys
            key = matched_points_plt.get_key() if matched_points_plt is not None else None
            if key == "" or key is None:
                key = err_plt.get_key() if err_plt is not None else None
            if key == "" or key is None:
                key = plt3d.get_key() if plt3d is not None else None

            # press 'q' to exit!
            key_cv = cv2.waitKey(1) & 0xFF
            if key == "q" or (key_cv == ord("q")):
                break
            if viewer3D and viewer3D.is_closed():
                break
        img_id += 1

    # print('press a key in order to exit...')
    # cv2.waitKey(0)

    # Post-processing: Build feature tracks for longevity visualization
    print("\n=== Building feature tracks for longevity analysis ===")
    if len(all_kps) > 0 and len(all_images) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for saving
            import matplotlib.pyplot as plt
            from tqdm import tqdm

            # Check if descriptors are missing (e.g., LK tracker doesn't compute them)
            descriptors_missing = all(des is None for des in all_des)

            if descriptors_missing:
                print("Descriptors not available (optical flow tracker). Computing ORB descriptors for longevity analysis...")
                orb = cv2.ORB_create(nfeatures=2000)

                for i in tqdm(range(len(all_images)), desc="Computing descriptors"):
                    if all_images[i] is not None and all_kps[i] is not None and len(all_kps[i]) > 0:
                        # Convert keypoints array to cv2.KeyPoint objects
                        kps_cv = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=20) for kp in all_kps[i]]
                        # Compute descriptors for these keypoints
                        _, des = orb.compute(all_images[i], kps_cv)
                        all_des[i] = des
                    else:
                        all_des[i] = None

            # Use BFMatcher for track building (NightHawk approach)
            # Determine correct norm based on descriptor type
            # Binary descriptors (ORB, BRISK, AKAZE) use HAMMING
            # Float descriptors (SIFT, SURF, ROOT_SIFT) use L2

            # Find first non-None descriptor to check type
            first_valid_des = None
            for des in all_des:
                if des is not None:
                    first_valid_des = des
                    break

            # If all descriptors are still None after fallback, we can't build tracks
            if first_valid_des is None:
                print("ERROR: No valid descriptors found. Cannot build tracks.")
                print("This can happen if:")
                print("  1. The tracker doesn't produce descriptors (e.g., LK)")
                print("  2. The fallback descriptor computation failed")
                print("  3. No features were detected in any frame")
                raise ValueError("Cannot build tracks without descriptors")

            if descriptors_missing or first_valid_des.dtype == np.uint8:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                print("Using NORM_HAMMING for binary descriptors")
            else:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                print("Using NORM_L2 for float descriptors")

            tracks = {}
            next_track_id = 0

            # Convert descriptors to proper type if needed
            for i in range(len(all_des)):
                if all_des[i] is not None:
                    # ORB descriptors are uint8, don't convert to float32
                    if all_des[i].dtype == np.uint8:
                        pass  # Keep as uint8 for binary descriptors
                    elif all_des[i].dtype != np.float32:
                        all_des[i] = all_des[i].astype(np.float32)

            # Build tracks sequentially frame-by-frame
            # Each feature gets ONE unique track ID for its entire lifetime
            # active_tracks: maps feature index in current frame -> track_id
            active_tracks = {}

            for frame_idx in tqdm(range(len(all_kps)), desc="Building tracks"):
                if all_kps[frame_idx] is None or all_des[frame_idx] is None:
                    active_tracks = {}  # Reset if frame is invalid
                    continue

                current_descriptors = all_des[frame_idx]
                num_features = len(all_kps[frame_idx])

                if frame_idx == 0:
                    # First frame: create new track for each feature
                    for i in range(num_features):
                        tracks[next_track_id] = [(frame_idx, i)]
                        active_tracks[i] = next_track_id
                        next_track_id += 1
                    continue

                # Get previous frame's descriptors
                prev_frame_idx = frame_idx - 1
                prev_descriptors = all_des[prev_frame_idx]

                if prev_descriptors is None or len(prev_descriptors) == 0 or len(active_tracks) == 0:
                    # No previous descriptors or tracks, start fresh
                    active_tracks = {}
                    for i in range(num_features):
                        tracks[next_track_id] = [(frame_idx, i)]
                        active_tracks[i] = next_track_id
                        next_track_id += 1
                    continue

                # Match previous frame to current frame
                matches = bf.knnMatch(prev_descriptors, current_descriptors, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                for match in matches:
                    if len(match) >= 2:
                        m, n = match[:2]
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                # Track which current features have been matched
                matched_current_features = set()
                new_active_tracks = {}

                for match in good_matches:
                    prev_feat_idx = match.queryIdx
                    curr_feat_idx = match.trainIdx

                    # Avoid double-matching the same current feature
                    if curr_feat_idx in matched_current_features:
                        continue

                    if prev_feat_idx in active_tracks:
                        # Extend existing track
                        track_id = active_tracks[prev_feat_idx]
                        tracks[track_id].append((frame_idx, curr_feat_idx))
                        new_active_tracks[curr_feat_idx] = track_id
                        matched_current_features.add(curr_feat_idx)

                # Create new tracks for unmatched features in current frame
                for i in range(num_features):
                    if i not in matched_current_features:
                        tracks[next_track_id] = [(frame_idx, i)]
                        new_active_tracks[i] = next_track_id
                        next_track_id += 1

                active_tracks = new_active_tracks

            # Calculate track statistics
            track_lengths = [len(track) for track in tracks.values()]
            mean_track_length = np.mean(track_lengths) if track_lengths else 0.0

            print(f"Total tracks created: {len(tracks)}")
            print(f"Mean track length: {mean_track_length:.2f} frames")

            # Generate longevity plot
            print("Generating longevity plot...")
            plt.figure(figsize=(24, 12))
            for track_id, track in tqdm(tracks.items(), desc="Plotting tracks"):
                frames = [f for f, _ in track]
                plt.plot([track_id] * len(frames), frames, marker='o', linestyle='-', lw=0.05)

            plt.xlabel('Feature Track ID')
            plt.ylabel('Frame ID')
            plt.title(f'Feature Tracks Over Multiple Frames - Mean Track Length: {mean_track_length:.2f}')
            plt.gca().invert_yaxis()  # Frame 0 at top

            # Determine sequence name from config for filename
            sequence_name = config.dataset_settings.get('name', 'unknown')
            if not os.path.exists(kResultsFolder):
                os.makedirs(kResultsFolder, exist_ok=True)

            longevity_plot_file = f"{kResultsFolder}/track_longevity_{sequence_name}.png"
            print(f"Saving track longevity plot to {longevity_plot_file}")
            plt.savefig(longevity_plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            # Generate histogram of track lengths
            print("Generating track length histogram...")
            plt.figure(figsize=(12, 8))
            max_length = max(track_lengths) if track_lengths else 1

            # Use 95th percentile as x-axis limit to focus on where most data is
            percentile_95 = np.percentile(track_lengths, 95) if track_lengths else max_length
            x_limit = int(min(max_length, max(percentile_95 * 1.2, mean_track_length * 3)))

            bins = np.arange(1, x_limit + 2, 1)  # Bins of width 1 frame, starting from 1
            plt.hist(track_lengths, bins=bins, edgecolor='black', alpha=0.7)
            plt.axvline(mean_track_length, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_track_length:.2f}')
            plt.xlabel('Track Length (frames)')
            plt.ylabel('Number of Tracks')
            plt.title(f'Distribution of Feature Track Lengths - {sequence_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim(1, x_limit)
            # Force integer ticks on x-axis
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            histogram_file = f"{kResultsFolder}/track_histogram_{sequence_name}.png"
            print(f"Saving track histogram to {histogram_file}")
            plt.savefig(histogram_file, dpi=150, bbox_inches='tight')
            plt.close()

            # Generate survival curve
            # For each track length X, calculate % of tracks that survived at least X frames
            print("Generating survival curve...")
            plt.figure(figsize=(12, 8))

            total_tracks = len(track_lengths)

            # Calculate survival: for each possible length, what % of tracks >= that length
            unique_lengths = np.arange(1, max_length + 1)
            survival_pct = []
            for length in unique_lengths:
                num_surviving = np.sum(np.array(track_lengths) >= length)
                survival_pct.append(100.0 * num_surviving / total_tracks)

            # Find where survival drops below 5% to set x-axis limit
            survival_arr = np.array(survival_pct)
            idx_5pct = np.where(survival_arr <= 5)[0]
            if len(idx_5pct) > 0:
                x_limit_survival = int(unique_lengths[idx_5pct[0]] * 1.2)
            else:
                x_limit_survival = max_length
            x_limit_survival = max(x_limit_survival, int(mean_track_length * 3))

            plt.plot(unique_lengths, survival_pct, linewidth=2, color='blue')
            plt.fill_between(unique_lengths, survival_pct, alpha=0.3)
            plt.xlabel('Track Age (frames)')
            plt.ylabel('% of Tracks Surviving')
            plt.title(f'Track Survival Curve - {sequence_name}\nMean: {mean_track_length:.2f}')
            plt.grid(True, alpha=0.3)
            plt.xlim(1, x_limit_survival)
            plt.ylim(0, 100)

            # Add annotation for 2 frames survival
            if len(survival_arr) >= 2:
                pct_at_2 = survival_arr[1]  # Index 1 = 2 frames (since unique_lengths starts at 1)
                plt.axhline(pct_at_2, color='r', linestyle=':', alpha=0.5)
                plt.axvline(2, color='r', linestyle=':', alpha=0.5)
                plt.annotate(f'{pct_at_2:.1f}% survive 2 frames',
                           xy=(2, pct_at_2),
                           xytext=(2 + x_limit_survival*0.1, pct_at_2 + 5),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=10, color='red')

            # Force integer ticks on x-axis
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            survival_file = f"{kResultsFolder}/track_survival_{sequence_name}.png"
            print(f"Saving survival curve to {survival_file}")
            plt.savefig(survival_file, dpi=150, bbox_inches='tight')
            plt.close()

            print("=== Track longevity analysis complete ===\n")

        except Exception as e:
            print(f"Warning: Could not complete track longevity analysis: {e}")
            import traceback
            traceback.print_exc()

    if is_draw_traj_img:
        if not os.path.exists(kResultsFolder):
            os.makedirs(kResultsFolder, exist_ok=True)
        print(f"saving {kResultsFolder}/map.png")
        cv2.imwrite(f"{kResultsFolder}/map.png", traj_img)

    # Save track longevity data and statistics
    if hasattr(vo, 'track_manager'):
        track_data = vo.track_manager.export_tracks()
        stats = track_data['statistics']

        # Determine sequence name from config for filename
        sequence_name = config.dataset_settings.get('name', 'unknown')
        track_data_file = f"{kResultsFolder}/tracks_{sequence_name}.json"

        print(f"Saving track data to {track_data_file}")
        with open(track_data_file, 'w') as f:
            json.dump(track_data, f, indent=2)

        # Print summary statistics
        print("\n=== Feature Tracking Statistics ===")
        print(f"Total tracks: {stats['total_tracks']}")
        print(f"Mean track length: {stats['mean_track_length']:.2f} frames")
        print(f"Median track length: {stats['median_track_length']:.2f} frames")
        print(f"Std track length: {stats['std_track_length']:.2f} frames")
        print(f"Max track length: {stats['max_track_length']} frames")
        print(f"Min track length: {stats['min_track_length']} frames")
        print(f"Total frames processed: {track_data['total_frames']}")
        print("===================================\n")


    if plt3d:
        plt3d.quit()
    if viewer3D:
        viewer3D.quit()
    if err_plt:
        err_plt.quit()
    if matched_points_plt:
        matched_points_plt.quit()

    if not args.headless:
        cv2.destroyAllWindows()
