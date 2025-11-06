import numpy as np

from scipy.spatial.transform import Rotation as R


def euler_to_matrix(euler_angles: np.ndarray, convention: str = 'xyz') -> np.ndarray:
    is_single = euler_angles.ndim == 1
    if is_single:
        euler_angles = euler_angles[np.newaxis, :]
    rot = R.from_euler(convention, euler_angles)
    return rot


def add_delta_to_euler_pose(base_euler, delta_axis_angle):
    base_rot = euler_to_matrix(base_euler)
    delta_rot = R.from_rotvec(delta_axis_angle)
    target_rot = delta_rot * base_rot
    target_euler = target_rot.as_euler('xyz')
    return target_euler
