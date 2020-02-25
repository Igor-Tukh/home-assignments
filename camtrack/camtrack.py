#! /usr/bin/env python3
import cv2

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    pose_to_view_mat3x4,
    rodrigues_and_translation_to_view_mat3x4
)

"""
Params section:

1) Triangulation params:
   A) Max projection error
   B) Min triangulation angle deg
   c) Min depth
"""

MAX_PROJECTION_ERROR = 1.
MIN_TRIANGULATION_ANGLE_DEG = .007
MIN_DEPTH = .001

TRIANGULATION_PARAMS = TriangulationParameters(max_reprojection_error=MAX_PROJECTION_ERROR,
                                               min_triangulation_angle_deg=MIN_TRIANGULATION_ANGLE_DEG,
                                               min_depth=MIN_DEPTH)


def add_new_points(frames, view_mats, frame_ind_1, frame_ind_2, all_points, intrinsic_mat):
    correspondences = build_correspondences(frames[frame_ind_1], frames[frame_ind_2])
    points, ids, _ = triangulate_correspondences(correspondences,
                                                 view_mats[frame_ind_1],
                                                 view_mats[frame_ind_2],
                                                 intrinsic_mat, TRIANGULATION_PARAMS)

    new_points_number = 0
    for point, point_id in zip(points, ids):
        if point_id not in all_points or all_points[point_id] is None:
            all_points[point_id] = point
            new_points_number += 1

    print(f'Frames: {frame_ind_1} and {frame_ind_2}, found {new_points_number} new points')


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats_tmp = [None for _ in range(len(corner_storage))]
    all_points = {}
    not_none_mats = [known_view_1[0], known_view_2[0]]
    none_mats = set([ind for ind in range(len(view_mats_tmp)) if ind not in not_none_mats])

    view_mats_tmp[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats_tmp[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    add_new_points(corner_storage, view_mats_tmp, known_view_1[0], known_view_2[0], all_points, intrinsic_mat)

    old_len = 2
    for _ in range(len(view_mats_tmp)):
        for ind in none_mats:
            corner_ids = list(corner_storage[ind].ids.reshape(-1))
            intersection_inds = [point_ind for point_ind in corner_ids
                                 if point_ind in all_points and all_points[point_ind] is not None]
            # There is an assert, so points number has to be at least 5
            if len(intersection_inds) < 5:
                continue
            # print(ind, len(corner_storage), len(corner_storage[ind].points), np.max(intersection_inds))
            corner_points = {i: p for i, p in zip(corner_ids, corner_storage[ind].points)}
            intersection_points_c = np.array([corner_points[i] for i in intersection_inds])
            intersection_points_f = np.array([all_points[i] for i in intersection_inds])
            # print(intersection_points_f)
            # print(intersection_points_f.shape, intersection_points_c.shape, intrinsic_mat)
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(intersection_points_f, intersection_points_c,
                                                             intrinsic_mat, None)

            if not retval:
                continue

            print(f'Processing frame: {ind}')

            newly_none_number = 0
            for i in intersection_inds:
                if i not in inliers:
                    newly_none_number += 1
                    all_points[i] = None

            print(f'{newly_none_number} points filled as None, len of inliers: {inliers.shape[0]}')
            print(f'Number of not points: {len([p for p in all_points.keys() if all_points[p] is not None])}')

            view_mats_tmp[ind] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

            for not_none_ind in not_none_mats:
                add_new_points(corner_storage, view_mats_tmp, ind, not_none_ind, all_points, intrinsic_mat)

            not_none_mats.append(ind)
            none_mats.remove(ind)
            break

        if len(not_none_mats) == old_len:
            break
        old_len = len(not_none_mats)

    view_mats = [None for _ in range(len(corner_storage))]
    for view_mat_ind in not_none_mats:
        view_mats[view_mat_ind] = view_mats_tmp[view_mat_ind]
    last_ind = 0
    for i in range(len(view_mats)):
        if view_mats[i] is None:
            view_mats[i] = view_mats[last_ind]
        else:
            last_ind = i
    all_points = {k: v for k, v in all_points.items() if v is not None}
    point_cloud_builder = PointCloudBuilder(np.array(list(all_points.keys()), dtype=np.int),
                                            np.array(list(all_points.values())))

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # ./camtrack.py ../../dataset/fox_head_short/rgb.mov ../data_examples/fox_camera.yml track.yml point_cloud.yml --camera-poses ../data_examples/fox_track.yml
    # python render.py ..\data_examples\fox_camera.yml ..\data_examples\fox_track.yml ..\data_examples\fox_point_cloud_with_colors.yml

    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
