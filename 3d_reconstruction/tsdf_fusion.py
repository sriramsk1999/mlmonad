import numpy as np
import utils


class TSDFVolume:
    def __init__(self, resolution, max_depth, truncation_threshold, intrinsics):
        self.grid_shape = (resolution, resolution, resolution)
        self.tsdf_volume = np.ones(self.grid_shape, dtype=np.float32)
        self.weight_volume = np.zeros(self.grid_shape, dtype=np.int16)
        # Flattened list of indices for every voxel (N^3, 3)
        self.voxel_idxs = np.indices(self.grid_shape).reshape(3, -1).T
        # We compute the TSDF at the center of each voxel.
        # Additionally we scale down the size of the voxels as our depth values are small.
        self.voxel_coords = (self.voxel_idxs.astype(np.float32) + 0.5) * 0.02
        self.intrinsics = intrinsics
        self.max_depth = max_depth
        self.truncation_threshold = truncation_threshold

    def integrate(self, depth: np.ndarray, pose: np.ndarray) -> np.ndarray:
        transformed_pcd, pix_coord = self.project_voxels_to_pixels(pose)
        mask, valid_depth = self.get_valid_mask(pix_coord, depth)
        self.update_tsdf(transformed_pcd, mask, valid_depth)

    def project_voxels_to_pixels(self, pose):
        # Bring voxel coords to current camera frame
        transformed_coords = utils.transform_point_cloud(self.voxel_coords, pose)

        # Apply the camera intrinsics to project points to 2D
        pix_coord_homogeneous = self.intrinsics @ transformed_coords.T  # Shape (3, N)

        # Normalize to get pixel coordinates
        pix_coord = pix_coord_homogeneous[:2, :] / pix_coord_homogeneous[2, :]
        pix_coord = np.round(pix_coord).astype(int).T
        return transformed_coords, pix_coord

    def get_valid_mask(self, pix_coord, depth):
        height, width = depth.shape
        # mask values beyond threshold
        depth[depth > self.max_depth] = 0

        pix_u, pix_v = pix_coord[:, 0], pix_coord[:, 1]
        # Valid pixels lie within the bounds of the image
        mask = np.ones_like(pix_u).astype(bool)
        mask &= pix_u >= 0
        mask &= pix_u < width
        mask &= pix_v >= 0
        mask &= pix_v < height

        valid_depth = depth[pix_v[mask], pix_u[mask]]
        return mask, valid_depth

    def update_tsdf(self, voxel_pcd, valid_pix_mask, depth):
        voxel_pcd = voxel_pcd[valid_pix_mask]
        voxel_z = voxel_pcd[:, 2]
        sdf = depth - voxel_z

        # Only keep -threshold < SDF < threshold
        valid_sdf_mask = depth > 0
        valid_sdf_mask &= sdf > -self.truncation_threshold
        valid_sdf_mask &= sdf < self.truncation_threshold

        # Calculate the TSDF
        tsdf = sdf[valid_sdf_mask] / self.truncation_threshold
        # Assume a constant weight of 1 for new observations
        weight = 1
        # A bit convoluted, but we're just using 2 masks to get the relevant voxels
        vx, vy, vz = self.voxel_idxs[valid_pix_mask][valid_sdf_mask].T

        tsdf_old = self.tsdf_volume[vx, vy, vz].astype(np.float32)
        tsdf_new = tsdf
        w_old = self.weight_volume[vx, vy, vz]
        w_new = w_old + weight
        tsdf_vol_new = (w_old * tsdf_old + weight * tsdf_new) / w_new

        self.tsdf_volume[vx, vy, vz] = tsdf_vol_new
        self.weight_volume[vx, vy, vz] = w_new
