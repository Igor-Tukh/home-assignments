#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    """
    Params section:

    0) Common parameters
    """

    MAX_CORNERS = 2000
    EPS = 1e-9
    MAX_SHIFT = 1.

    """
    1) Shi-Tomasi method:
        a) Number of strongest corners to detect
        b) Quality level (0..1). All corners with quality below it will be ignored
        c) Minimum euclidean distance between corners detected
    """

    N_STRONGEST = 1500
    QULITY_LEVEL = 0.11
    MIN_DISTANCE = 15

    get_corners = lambda img: cv2.goodFeaturesToTrack(image_0, N_STRONGEST, QULITY_LEVEL, MIN_DISTANCE)

    """
    2) Lucas-Kanade optical Flow:
        a) Size of the search window at each pyramid level
        b) Maximal pyramid level (0-based, number of layers - 1)
        c) Termination criterion
    """

    WIN_SIZE = (10, 10)
    MAX_LEVEL = 2
    CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    get_next_points = lambda first_frame, second_frame, first_frame_corners: \
        cv2.calcOpticalFlowPyrLK(first_frame,
                                 second_frame,
                                 first_frame_corners,
                                 winSize=WIN_SIZE,
                                 maxLevel=MAX_LEVEL,
                                 criteria=CRITERIA,
                                 nextPts=None)
    """
    End of params section.
    
    Some initial parameter values have been collected from the corresponding opencv guide.
    """

    frame_sequence = [(frame * 255).astype(np.uint8) for frame in frame_sequence]

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    image_0 = frame_sequence[0]
    corners_0 = get_corners(image_0)
    n_corners = corners_0.shape[0]

    ids = np.arange(n_corners)
    points = corners_0.squeeze()
    sizes = np.full(n_corners, 5)
    corners = FrameCorners(ids, points, sizes)

    builder.set_corners_at_frame(0, corners)

    id_counter = n_corners

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        points_1, st, err = get_next_points(image_0, image_1, points)
        points_1 = points_1.squeeze()
        mask = st.squeeze() == 1

        # Additional check
        points_0, st, err = get_next_points(image_1, image_0, points_1)
        mask = mask & (st.squeeze() == 1)
        mask = mask & (np.linalg.norm(points_0 - points, axis=1) < MAX_SHIFT)

        ids = ids[mask]
        points = points_1[mask]
        sizes = sizes[mask]

        n_corners = ids.shape[0]

        corners_1 = get_corners(image_1)
        new_points = corners_1.squeeze()
        added = np.zeros(new_points.shape[0], dtype=bool)

        # Heuristics: let's iteratively select the furthest new corner.
        while n_corners < MAX_CORNERS:
            max_min_dist = None
            next_point_ind = -1
            for ind, new_point in enumerate(new_points):
                if added[ind]:
                    continue

                min_dist = None
                for point in points:
                    dist = np.linalg.norm(point - new_point)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist

                if min_dist is not None and min_dist < MIN_DISTANCE + EPS:
                    continue

                if min_dist is None or max_min_dist is None or max_min_dist < min_dist:
                    max_min_dist = min_dist
                    next_point_ind = ind

            if next_point_ind == -1:
                break

            ids = np.concatenate([ids, [id_counter]])
            points = np.concatenate([points, [new_points[next_point_ind]]])
            sizes = np.concatenate([sizes, [5]])

            added[next_point_ind] = True

            n_corners += 1
            id_counter += 1

        corners = FrameCorners(ids, points, sizes)
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
