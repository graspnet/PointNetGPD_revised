__author__ = 'chenxi-wang'
__version__ = '1.0' 

import os
import sys
import subprocess
import argparse
import logging
import time
import scipy.io as scio
# Must be imported before
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graspnetAPI import GraspNet
from utils import CameraInfo, generate_views, transform_points, batch_viewpoint_params_to_matrix, create_point_cloud_from_depth_image

# some params
V = 300   # number of views
A = 12      # number of gripper rotation angles
H = 0.02     # height of gripper
height = 0.02     # height of gripper
depth_base = 0.02
GRASP_MAX_WIDTH = 0.1
GRASP_MAX_DEPTH = 0.04

class GraspNetDataset(Dataset):
    def __init__(self, root, camera='kinect', split='train', num_points=20000, voxel_size=0.005, remove_outlier=True):
        assert(num_points is None or num_points<=50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.remove_outlier = remove_outlier
        self.num_sample = 10
        self.dist_thresh = 0.01
        self.score_thresh = 0.11

        graspnet = GraspNet(root, camera=camera, split=split)
        self.rgbpath, self.depthpath, self.labelpath, self.metapath, self.scenename = graspnet.loadData()
        self.grasp_labels = graspnet.loadGraspLabels(retrun_collision=True)

        self.num_views, self.num_angles, self.num_depths = 300, 12, 4
        self.viewpoints = generate_views(self.num_views)


    def __getitem__(self, index):
        
        start = time.time()
        rgb = np.array(Image.open(self.rgbpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))

        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])  # 3,4,9
        scene = self.scenename[index]
        try:
            cls_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            obj_idxs = cls_idxs-1
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        # realsense = CameraInfo(1280.0, 720.0, 927.17, 927.37, 651.32, 349.62, 1000.0)
        # kinect = CameraInfo(1280.0, 720.0, 631.54864502, 631.20751953, 638.43517329, 366.49904066, 1000.0)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera)
        t0 = time.time()
        # print('cloud loading time: ', t0-start )
        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            mask = (depth_mask & seg_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        rgb_masked = rgb[mask]
        seg_masked = seg[mask]

        # sample points
        if self.num_points is not None:
            if len(cloud_masked) >= self.num_points:
                idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
        else:
            idxs = np.arange(len(cloud_masked))
        
        cloud_sampled = cloud_masked[idxs].astype(np.float32)
        rgb_sampled = rgb_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy().astype(np.int64)
        
        t1 = time.time()
        # print('cloud sampling time: ', t1-t0)
        feats = []
        rgbs = []
        scores = []
        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == cls_idxs[i]).sum() < 50:
                continue
            # get grasp labels
            grasp_points, grasp_offsets, grasp_scores, collision_label = self.grasp_labels[obj_idx]
            grasp_scores[collision_label] = -2
            normals = estimate_normals(grasp_points, k=10, align_direction=False, ret_cloud=False)

            # get point cloud in scene
            cloud_obj = cloud_sampled[np.where(objectness_label==cls_idxs[i])]
            rgb_obj = rgb_sampled[np.where(objectness_label==cls_idxs[i])]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(cloud_obj)
            pc.colors = o3d.utility.Vector3dVector(rgb_obj)
            downpc = pc.voxel_down_sample(voxel_size=self.voxel_size)
            cloud_obj = np.array(downpc.points).astype(np.float32)
            rgb_obj = np.array(downpc.colors).astype(np.float32)

            # get object pose
            obj_pose = np.zeros([4,4])
            obj_pose[3,3] = 1
            obj_pose[:3,:] = poses[:,:,i]

            # get visible grasp points
            cloud_rotated = transform_points(cloud_obj, np.linalg.inv(obj_pose))
            dists = np.linalg.norm(grasp_points[:,np.newaxis]-cloud_rotated[np.newaxis,:], axis=-1)
            dists_mask = (dists.min(axis=-1)<self.dist_thresh)
            grasp_points = grasp_points[dists_mask]
            normals = normals[dists_mask]
            grasp_offsets = grasp_offsets[dists_mask]
            grasp_scores = grasp_scores[dists_mask]
            collision_label = collision_label[dists_mask]

            # sample grasps
            grasp_points, grasp_viewpoints, grasp_offsets, grasp_scores = self.sample_grasps(grasp_points, normals, grasp_offsets, grasp_scores, collision_label)

            # crop points
            cloud_centered = cloud_rotated[np.newaxis,:,:] - grasp_points[:,np.newaxis,:]
            grasp_angles = grasp_offsets[:, 0]
            grasp_depths = grasp_offsets[:, 1]
            grasp_widths = grasp_offsets[:, 2]
            grasp_poses = batch_viewpoint_params_to_matrix(-grasp_viewpoints, grasp_angles)
            targets = np.matmul(cloud_centered, grasp_poses) #(2*num_sample, num_point, 3)

            for ind in range(targets.shape[0]):
                target = targets[ind]
                mask1 = ((target[:,2]>-height) & (target[:,2]<height))
                mask2 = ((target[:,0]<grasp_depths[ind]) & (target[:,0]>-depth_base))
                mask3 = ((target[:,1]>-grasp_widths[ind]/2) & (target[:,1]<grasp_widths[ind]/2))
                mask = (mask1 & mask2 & mask3)
                pc = target[mask]
                if len(pc) == 0:
                    continue
                feats.append(pc)
                rgbs.append(rgb_obj[mask])
                scores.append(grasp_scores[ind])

        return feats, rgbs, scores

    def sample_grasps(self, points, normals, offsets, scores, collision_label):
        N, V, A, D, _ = offsets.shape
        # select viewpoints by normal
        viewpoints_ = self.viewpoints[np.newaxis,:,:]
        normals_ = normals[:,np.newaxis,:]
        dists = np.linalg.norm(normals_-viewpoints_, axis=-1)
        view_inds = np.argmin(dists, axis=-1)
        viewpoints = self.viewpoints[view_inds] #(Np, 3)
        offsets = offsets[np.arange(N),view_inds] #(Np, A, D, 3)
        scores = scores[np.arange(N),view_inds] #(Np, A, D)
        collision = collision_label[np.arange(N),view_inds] #(Np, A, D)

        # estimate darboux frame
        frames = estimate_darboux_frame(points, normals, points, normals, self.dist_thresh)

        # select angles by darboux frame
        viewpoints_ = np.tile(viewpoints[:,np.newaxis,:], [1,A,1]).reshape([-1,3])
        angles_ = offsets[:,:,0,0].reshape(-1)
        R1_ = batch_viewpoint_params_to_matrix(-viewpoints_, angles_).reshape([N,A,3,3])
        R2_ = batch_viewpoint_params_to_matrix(-viewpoints_, angles_+np.pi).reshape([N,A,3,3])
        diff1 = np.matmul(frames[:,np.newaxis], np.transpose(R1_,[0,1,3,2]))
        diff2 = np.matmul(frames[:,np.newaxis], np.transpose(R2_,[0,1,3,2]))
        trace1 = np.trace(diff1, axis1=2, axis2=3)
        trace2 = np.trace(diff2, axis1=2, axis2=3)
        delta1 = np.arccos(np.clip((trace1-1)/2, -1, 1)) #(Np,A)
        delta2 = np.arccos(np.clip((trace2-1)/2, -1, 1)) #(Np,A)
        delta = np.minimum(delta1, delta2) #(Np,A)
        angle_indices = delta.argmin(axis=-1)

        offsets = offsets[np.arange(N), angle_indices] #(Np,D,3)
        scores = scores[np.arange(N), angle_indices] #(Np,D)
        collision = collision[np.arange(N), angle_indices] #(Np,D)

        # remove collision
        points = np.tile(points[:,np.newaxis,:], [1,D,1])
        viewpoints = np.tile(viewpoints[:,np.newaxis,:], [1,D,1])
        points = points[~collision] #(-1, 3)
        viewpoints = viewpoints[~collision] #(-1, 3)
        offsets = offsets[~collision] #(-1, 3)
        scores = scores[~collision] #(-1)

        # sample data
        pos_mask = ((scores>0) & (scores<self.score_thresh))
        pos_index = np.where(pos_mask)[0]
        neg_index = np.where(~pos_mask)[0]
        # print(pos_index.shape, neg_index.shape)
        num_sample = min(self.num_sample, len(pos_index))
        num_sample = min(num_sample, len(neg_index))
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)
        pos_index = pos_index[:num_sample]
        neg_index = neg_index[:num_sample]
        sample_index = np.concatenate([pos_index,neg_index])
        # print(sample_index.shape)

        points = points[sample_index] #(2*num_sample, 3)
        viewpoints = viewpoints[sample_index] #(2*num_sample, 3)
        offsets = offsets[sample_index] #(2*num_sample, 3)
        scores = scores[sample_index] #(2*num_sample)
        return points, viewpoints, offsets, scores

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)


def estimate_normals(points, k=10, align_direction=False, ret_cloud=False):
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=k)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(search_param=search_param)
    if align_direction:
        cloud.orient_normals_to_align_with_direction([-1,0,0])
    if ret_cloud:
        return cloud
    else:
        normals = np.array(cloud.normals)
        return normals

def estimate_darboux_frame(target_points, target_normals, ref_points, ref_normals, dist_thresh):
    ''' Estimate a set of darboux frames for a set of target points given reference points.
        Input:
            target_points: [numpy.ndarray, (Nt,3), numpy.float32]
                target points to estimate
            target_normals: [numpy.ndarray, (Nt,3), numpy.float32]
                normals of target points
            ref_points: [numpy.ndarray, (Nr,3), numpy.float32]
                reference points to provide neighbours for target points
            ref_normals: [numpy.ndarray, (Nr,3), numpy.float32]
                normals of reference points
            dist_thresh: [float32]
                any points within [dist_thresh] to a target point will be treated as neighbours
        Output:
            frames: [numpy.ndarray, (Nt,3,3), numpy.float32]
    '''
    Nt = target_points.shape[0]
    dists = np.linalg.norm(target_points[:,np.newaxis,:]-ref_points[np.newaxis,:,:], axis=-1)
    nn_mask = (dists <= dist_thresh)
    nn_normals = np.tile(ref_normals[np.newaxis,:,:],[Nt,1,1]) #(Nt,Nr,3)
    nn_normals[~nn_mask] = 0
    nn_normals_ = np.transpose(nn_normals, [0,2,1]) #(Nt,3,Nr)
    Ms = np.matmul(nn_normals_, nn_normals) #(Nt,3,3)
    eigenvalues, eigenvectors = np.linalg.eig(Ms)
    eigenvalues = eigenvalues.real #(Nt,3)
    eigenvectors = eigenvectors.real #(Nt,3,3)

    min_indices = eigenvalues.argmin(axis=-1)
    max_indices = eigenvalues.argmax(axis=-1)
    axes_x = eigenvectors[np.arange(Nt),:,max_indices]
    axes_z = eigenvectors[np.arange(Nt),:,min_indices]

    dir_mask = (np.sum(target_normals*axes_x,axis=-1)>0)
    axes_x[dir_mask] *= -1
    axes_y = np.cross(axes_z, axes_x)
    frames = np.stack([axes_x,axes_y,axes_z], axis=2) #(Nt,3,3)

    return frames

def gen_single_view_dataset(root, split, savedir):
    savedir = os.path.join(savedir, split)
    labelpath = os.path.join(savedir, "labels_%s.npy" % split)
    cloudsavedir = os.path.join(savedir, "cloud")
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    if not os.path.exists(cloudsavedir):
        os.mkdir(cloudsavedir)
    dataset = GraspNetDataset(root, camera='kinect', split=split, num_points=None, voxel_size=0.002, remove_outlier=True)

    overall_scores = list()
    total_cnt = 0

    for (feats, rgbs, scores) in tqdm(dataset):
        for (feat, rgb) in zip(feats, rgbs):
            pc = o3d.geometry.PointCloud()
            pc = estimate_normals(feat, k=20, align_direction=True, ret_cloud=True)
            pc.colors = o3d.utility.Vector3dVector(rgb)
            cloudsavepath = os.path.join(cloudsavedir, "%06d.ply" % total_cnt)
            o3d.io.write_point_cloud(cloudsavepath, pc, write_ascii=False, compressed=True)
            total_cnt += 1
        overall_scores += scores

    np.save(labelpath, np.array(overall_scores))


if __name__ == '__main__':
    root = '/data/Benchmark/graspnet'
    # split = 'train1'
    # split = 'train2'
    # split = 'train3'
    # split = 'train4'
    split = 'test_seen'
    # split = 'test_similar'
    # split = 'test_novel'
    savedir = '/data/chenxi/workspace/graspnetAPI/gpd_data_v4'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    gen_single_view_dataset(root, split, savedir)
