"""
PackNet-SfM datasets
====================

These datasets output images, camera calibration, depth maps and poses for depth and pose estimation

- KITTIDataset: reads from KITTI_raw
- DGPDataset: reads from a DGP .json file
- ImageDataset: reads from a folder containing image sequences (no support for depth maps)

"""
import sys
sys.path.append('/home/user/nvme4/elice2/cloud2/DiffusionDriveForward/external/packnet_sfm')
sys.path.append('/home/user/nvme4/elice2/cloud2/DiffusionDriveForward/external/dgp')
