import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=True,
    help="path to dataset",
)

args = parser.parse_args()

N = len(os.listdir(f"{args.dataset}/rgb/"))
K = utils.load_intrinsics(f"{args.dataset}/cameras.txt")
poses = utils.load_pose(f"{args.dataset}/images.txt")

for i in tqdm(range(1, N + 1)):
    rgb = cv2.cvtColor(
        cv2.imread(f"{args.dataset}/rgb/img_{i:05}.png"), cv2.COLOR_BGR2RGB
    )
    dep = np.load(f"{args.dataset}/depth/img_{i:05}.npy")
    pose = poses[i]
