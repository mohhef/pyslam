"""
Track Manager for maintaining persistent feature track information across frames.

This module provides functionality to track individual features over multiple frames,
maintaining their unique IDs, lifetimes, and history.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class TrackManager:
    """
    Manages persistent feature tracks across multiple frames.

    Maintains unique track IDs for features and tracks their lifetimes,
    enabling analysis of feature tracking performance.
    """

    def __init__(self):
        """Initialize the track manager."""
        self.tracks = {}  # track_id -> track_info dict
        self.next_track_id = 0  # Counter for assigning new track IDs
        self.current_frame = -1  # Current frame number
        self.active_track_ids = []  # List of track IDs active in current frame
        self.ref_track_ids = []  # Track IDs from reference (previous) frame

    def update(self, idxs_ref: Optional[np.ndarray], idxs_cur: Optional[np.ndarray],
               kps_cur: np.ndarray, frame_id: int):
        """
        Update tracks based on frame-to-frame matches.

        Args:
            idxs_ref: Indices of matched features in reference frame (None if first frame)
            idxs_cur: Indices of matched features in current frame (None if first frame)
            kps_cur: Current frame keypoints (Nx2 array)
            frame_id: Current frame ID
        """
        self.current_frame = frame_id

        # Handle None or empty keypoints
        if kps_cur is None or len(kps_cur) == 0:
            return

        # First frame: initialize all features as new tracks
        if idxs_ref is None or idxs_cur is None or len(self.ref_track_ids) == 0:
            self._initialize_tracks(kps_cur, frame_id)
            return

        # Handle empty match arrays
        if len(idxs_ref) == 0 or len(idxs_cur) == 0:
            # No matches - all current features are new tracks
            self._initialize_tracks(kps_cur, frame_id)
            return

        # Create new active track list
        new_active_track_ids = [-1] * len(kps_cur)

        # Map matched features to existing tracks
        for i, (idx_ref, idx_cur) in enumerate(zip(idxs_ref, idxs_cur)):
            # Bounds checking for both indices
            if idx_ref < len(self.ref_track_ids) and idx_cur < len(kps_cur):
                # Feature matched: continue existing track
                track_id = self.ref_track_ids[idx_ref]
                new_active_track_ids[idx_cur] = track_id

                # Update track info
                if track_id in self.tracks:
                    self.tracks[track_id]['age'] += 1
                    self.tracks[track_id]['last_frame'] = frame_id
                    self.tracks[track_id]['positions'].append(
                        (float(kps_cur[idx_cur][0]), float(kps_cur[idx_cur][1]))
                    )

        # Assign new track IDs to unmatched features
        for i, track_id in enumerate(new_active_track_ids):
            if track_id == -1:
                # New feature: create new track
                new_track_id = self._create_track(kps_cur[i], frame_id)
                new_active_track_ids[i] = new_track_id

        # Mark tracks that were lost (not in new_active_track_ids)
        lost_track_ids = set(self.ref_track_ids) - set(new_active_track_ids)
        for track_id in lost_track_ids:
            if track_id in self.tracks and track_id != -1:
                self.tracks[track_id]['is_active'] = False

        # Update reference for next frame
        self.ref_track_ids = new_active_track_ids
        self.active_track_ids = [tid for tid in new_active_track_ids if tid != -1]

    def _initialize_tracks(self, kps: np.ndarray, frame_id: int):
        """Initialize tracks for the first frame."""
        self.ref_track_ids = []
        self.active_track_ids = []

        for kp in kps:
            track_id = self._create_track(kp, frame_id)
            self.ref_track_ids.append(track_id)
            self.active_track_ids.append(track_id)

    def _create_track(self, kp: np.ndarray, frame_id: int) -> int:
        """
        Create a new track.

        Args:
            kp: Keypoint position (x, y)
            frame_id: Frame where track starts

        Returns:
            track_id: Unique ID for the new track
        """
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = {
            'age': 1,
            'first_frame': frame_id,
            'last_frame': frame_id,
            'positions': [(float(kp[0]), float(kp[1]))],
            'is_active': True
        }

        return track_id

    def get_active_track_count(self) -> int:
        """Get number of currently active tracks."""
        return len(self.active_track_ids)

    def get_total_track_count(self) -> int:
        """Get total number of tracks created."""
        return len(self.tracks)

    def get_track_ages(self) -> List[int]:
        """Get ages of all tracks."""
        return [track['age'] for track in self.tracks.values()]

    def get_active_track_ages(self) -> List[int]:
        """Get ages of currently active tracks."""
        return [self.tracks[tid]['age'] for tid in self.active_track_ids if tid in self.tracks]

    def get_average_track_length(self) -> float:
        """Get average track length across all tracks."""
        ages = self.get_track_ages()
        return np.mean(ages) if ages else 0.0

    def get_track_statistics(self) -> Dict:
        """
        Get comprehensive track statistics.

        Returns:
            Dictionary with track statistics
        """
        ages = self.get_track_ages()
        active_ages = self.get_active_track_ages()

        stats = {
            'total_tracks': len(self.tracks),
            'active_tracks': len(self.active_track_ids),
            'mean_track_length': float(np.mean(ages)) if ages else 0.0,
            'median_track_length': float(np.median(ages)) if ages else 0.0,
            'std_track_length': float(np.std(ages)) if ages else 0.0,
            'max_track_length': int(np.max(ages)) if ages else 0,
            'min_track_length': int(np.min(ages)) if ages else 0,
            'mean_active_track_length': float(np.mean(active_ages)) if active_ages else 0.0,
        }

        return stats

    def get_tracks_for_visualization(self) -> List[Tuple[int, int, int, int]]:
        """
        Get track data formatted for visualization.

        Returns:
            List of tuples: (track_id, first_frame, last_frame, age)
        """
        vis_data = []
        for track_id, track_info in self.tracks.items():
            vis_data.append((
                track_id,
                track_info['first_frame'],
                track_info['last_frame'],
                track_info['age']
            ))

        # Sort by first_frame for better visualization
        vis_data.sort(key=lambda x: x[1])

        return vis_data

    def export_tracks(self) -> Dict:
        """
        Export all track data for saving to file.

        Returns:
            Dictionary with complete track information
        """
        export_data = {
            'tracks': {},
            'statistics': self.get_track_statistics(),
            'total_frames': self.current_frame + 1
        }

        for track_id, track_info in self.tracks.items():
            export_data['tracks'][str(track_id)] = {
                'age': track_info['age'],
                'first_frame': track_info['first_frame'],
                'last_frame': track_info['last_frame'],
                'num_observations': len(track_info['positions']),
                'is_active': track_info['is_active']
            }

        return export_data
