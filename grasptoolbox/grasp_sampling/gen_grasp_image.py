__author__ = 'chenxi-wang'
__version__ = '1.0' 

import os
import cv2
from tqdm import tqdm
import numpy as np
import open3d as o3d

MAX_WIDTH = 0.15
HEIGHT = 0.02
MAX_DEPTH = 0.04
DEPTH_BASE = 0.02
IMAGE_SIZE = 90

def find_max_width(root, split):
    '''
    found max width: 0.15
    '''
    datadir = os.path.join(root, split)
    filenames = sorted(os.listdir(datadir))
    ymin, ymax = 0, 0
    max_width = 0
    for filename in tqdm(filenames):
        pc = o3d.io.read_point_cloud(os.path.join(datadir, filename))
        points = np.array(pc.points)
        ymin = min(ymin, points[:,1].min())
        ymax = max(ymax, points[:,1].max())
        max_width = max(max_width, points[:,1].max()-points[:,1].min())
    print('split: %s' % split)
    print('ymin: %.4f' % ymin)
    print('ymax: %.4f' % ymax)
    print('max width: %.4f' % max_width)

def generate_grasp_image(root, split):
    clouddir = os.path.join(root, split, 'cloud')
    imagedir = os.path.join(root, split, 'image')
    if not os.path.exists(imagedir):
        os.mkdir(imagedir)
    filenames = sorted(os.listdir(clouddir))
    
    for filename in tqdm(filenames):
        cloud = o3d.io.read_point_cloud(os.path.join(clouddir, filename))
        image = transform_cloud_to_image(cloud)
        imagesavepath = os.path.join(imagedir, '{}.jpg'.format(filename.split('.')[0]))
        cv2.imwrite(imagesavepath, image)

def transform_cloud_to_image(cloud):
    points = np.array(cloud.points)
    normals = np.array(cloud.normals)
    mask = (points[:,0]>-DEPTH_BASE)
    points = points[mask]
    normals = normals[mask]

    rows = ((points[:,0]+DEPTH_BASE) / (MAX_DEPTH+DEPTH_BASE))
    cols = (points[:,1] / MAX_WIDTH) + 0.5
    assert(np.all((rows>=0)&(rows<=1)))
    assert(np.all((cols>=0)&(cols<=1)))
    rows = ((1-rows) * (IMAGE_SIZE-1)).astype(np.int32)
    cols = (cols * (IMAGE_SIZE-1)).astype(np.int32)

    image = np.zeros([IMAGE_SIZE,IMAGE_SIZE,3], dtype=np.float32)
    for i,(r,c) in enumerate(zip(rows, cols)):
        if np.linalg.norm(image[r,c]) == 0:
            image[r][c] = np.abs(normals[i])
        else:
            image[r][c] += (np.abs(normals[i])-image[r][c]) / np.linalg.norm(image[r,c])

    kernel = np.ones([3,3], dtype=np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.normalize(image, image, 0., 1., cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = (image*255).astype(np.uint8)
    return image

if __name__ == "__main__":
    # split = 'train1'
    # split = 'train2'
    # split = 'train3'
    # split = 'train4'
    split = 'test_seen'
    # split = 'test_similar'
    # split = 'test_novel'
    root = '/data/chenxi/workspace/graspnetAPI'
    datadir = os.path.join(root, 'gpd_data_v4')
    generate_grasp_image(datadir, split)