import numpy as np
from transforms3d.quaternions import quat2mat


def load_pose(filename) -> np.ndarray:
    """
    Load the pose estimated by COLMAP from its images.txt and return as an ndarray.

    Args:
    - filename (str): Path to the file containing pose data.

    Returns:
    - np.ndarray: Array of camera-to-world transformation matrices (N, 4, 4)

    Assumes the input file format is structured in COLMAP's images.txt format:
    - Lines after the 4th line contain pose data.
    - Every second line starting from the 5th line contains relevant data.
    - Each line contains quaternion rotation (qw qx qy qz) followed by translation (tx ty tz).
    """
    with open(filename, "r") as f:
        data = f.readlines()
    data = data[4:]
    data = data[::2]
    camData = [i.strip().split()[1:8] for i in data]
    camData = np.array([[float(i) for i in row] for row in camData])
    translations = camData[:, 4:]
    rot_quat = camData[:, :4]
    rot_mat = np.array([quat2mat(i) for i in rot_quat])

    poses = np.array([np.eye(4) for _ in range(camData.shape[0])])
    poses[:, :3, :3] = rot_mat
    poses[:, :3, 3] = translations
    # COLMAP provides world-to-camera pose, invert to get camera-to-world
    poses = np.linalg.inv(poses)
    return poses


def load_intrinsics(filename) -> np.ndarray:
    """
    Load the intrinsics estimated by COLMAP from its cameras.txt and return as an ndarray.
    Assumes only one camera was used with the SIMPLE_RADIAL camera model.

    Args:
    - filename (str): Path to the file containing intrinsics.

    Returns:
    - np.ndarray: Camera intrinsics array (3, 3)

    Assumes the input file format is structured in COLMAP's cameras.txt format:
    - The 4th line contains intrinsics data.
    - The line contains camera idx, quaternion rotation (qw qx qy qz) followed by translation (tx ty tz).
    """
    with open(filename, "r") as f:
        data = f.readlines()
    data = data[3].strip().split()
    f, cx, cy = [float(i) for i in data[4:7]]
    K = np.eye(3)
    K[0, 0] = K[1, 1] = f
    K[0, 2], K[1, 2] = cx, cy
    return K
