from graspnetAPI.utils.utils import CameraInfo, create_point_cloud_from_depth_image, transform_points

import os
import sys
import time
import numpy as np
import open3d as o3d
from PIL import Image
import scipy.io as scio
from tqdm import tqdm

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h*w, 3])
        seg = seg.reshape(h*w)
    if trans is not None:
        cloud = transform_points(cloud, trans)
    foreground = cloud[seg>0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:,0] > xmin-outlier) & (cloud[:,0] < xmax+outlier))
    mask_y = ((cloud[:,1] > ymin-outlier) & (cloud[:,1] < ymax+outlier))
    mask_z = ((cloud[:,2] > zmin-outlier) & (cloud[:,2] < zmax+outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h,w])
    return workspace_mask

def load_cloud(scene_idx, frame_idx, graspnet_root, camera='kinect', remove_outlier=True):
    # load data
    scene_path = os.path.join(graspnet_root, 'scenes', 'scene_%04d' % scene_idx, camera)
    color = np.array(Image.open(os.path.join(scene_path, 'rgb', '%04d.png'%frame_idx))) / 255.0
    depth = np.array(Image.open(os.path.join(scene_path, 'depth', '%04d.png'%frame_idx)))
    seg = np.array(Image.open(os.path.join(scene_path, 'label', '%04d.png'%frame_idx)))
    meta = scio.loadmat(os.path.join(scene_path, 'meta', '%04d.mat'%frame_idx))
    # parse metadata
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camerainfo = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camerainfo, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    if remove_outlier:
        camera_poses = np.load(os.path.join(scene_path, 'camera_poses.npy'))
        align_mat = np.load(os.path.join(scene_path, 'cam0_wrt_table.npy'))
        trans = np.dot(align_mat, camera_poses[frame_idx])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)
    else:
        mask = depth_mask
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    return cloud_masked, color_masked
