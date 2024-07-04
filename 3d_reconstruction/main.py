import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

import utils
import tsdf_fusion

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="path to dataset",
)
parser.add_argument(
    "--tsdf_resolution",
    default=256,
    help="resolution of the tsdf volume",
)
parser.add_argument(
    "--max_depth",
    default=40,
    help="maximum depth to be considered. \
    NOTE: COLMAP depth is not metric, a value of 5 does not correspond to 5 metres.",
)
parser.add_argument(
    "--truncation_threshold",
    default=0.1,
    help="truncation threshold for the TSDF computation.",
)

args = parser.parse_args()

N = len(os.listdir(f"{args.dataset}/rgb/"))
K = utils.load_intrinsics(f"{args.dataset}/cameras.txt")
poses = utils.load_pose(f"{args.dataset}/images.txt")

volume = tsdf_fusion.TSDFVolume(
    args.tsdf_resolution, args.max_depth, args.truncation_threshold, K
)

for i in tqdm(range(1, N + 1)):
    rgb = cv2.cvtColor(
        cv2.imread(f"{args.dataset}/rgb/img_{i:05}.png"), cv2.COLOR_BGR2RGB
    )
    depth = np.load(f"{args.dataset}/depth/img_{i:05}.npy")
    pose = poses[i - 1]  # images are 1-indexed but pose is 0-indexed

    volume.integrate(depth, pose)
