# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import sys
import subprocess
import argparse
import logging
from time import time
# Must be imported before
try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')

import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler, RandomSampler
import torch.optim as optim
from torchvision.transforms import Compose as VisionCompose
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from scipy.linalg import expm, norm

from resnet import ResNet14, ResNet50

import MinkowskiEngine as ME

assert int(
    o3d.__version__.split('.')[1]
) >= 8, f'Requires open3d version >= 0.8, the current version is {o3d.__version__}'

parser = argparse.ArgumentParser()
parser.add_argument('--voxel_size', type=float, default=0.003)
parser.add_argument('--max_iter', type=int, default=120000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--stat_freq', type=int, default=50)
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--checkpoint_weights', type=str, default='graspnet.pth')
parser.add_argument('--weights', type=str, default='graspnet_sv.pth')
parser.add_argument('--load_optimizer', type=str, default='true')
config = parser.parse_args()

if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)

train_logger = logging.getLogger(__name__)
train_logger.setLevel(level = logging.INFO)
train_handler = logging.FileHandler("{}/train_log.txt".format(config.log_dir))
train_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
train_handler.setFormatter(formatter)
train_logger.addHandler(train_handler)

test_logger = logging.getLogger(__name__)
test_logger.setLevel(level = logging.INFO)
test_handler = logging.FileHandler("{}/test_log.txt".format(config.log_dir))
test_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
test_handler.setFormatter(formatter)
test_logger.addHandler(test_handler)

train_writer = SummaryWriter('{}/train'.format(config.log_dir))
test_writer = SummaryWriter('{}/test'.format(config.log_dir))

def log_string(logger, s):
    print(s)
    logger.info(s)

def collate_pointcloud_fn(list_data):
    coords_list = []
    feats_list = []
    labels_list = []
    for data in list_data: # list_data []
        coords_list.append(data[0])
        feats_list.append(data[1])
        labels_list.append(data[2])

    eff_num_batch = len(coords_list)
    assert len(labels_list) == eff_num_batch
    coords_batch = ME.utils.batched_coordinates(coords_list)
    feats_batch = torch.from_numpy(np.vstack(feats_list)).float()

    # Concatenate all lists
    return {
        'coords': coords_batch,             # [all_points, 3+batch_id]
        'feats': feats_batch,               # [all_points, 3]
        'labels': torch.LongTensor(labels_list), # [batch_id, 1]
    }

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

class GraspNetSVDataset(Dataset):
    def __init__(self, root, splits=['train1'], th=0.11, voxel_size=0.003, balance_data=False):
        self.root = root
        self.voxel_size = voxel_size
        self.datapath = []
        self.labels = []
        for split in splits:
            print('Loading {} set...'.format(split))
            label = np.load(os.path.join(root, split, 'labels_{}.npy'.format(split)))
            label = ((label>0) & (label<th)).astype(np.int64)
            self.labels.append(label)
            filenames = sorted(os.listdir(os.path.join(root, split, 'cloud')))
            for filename in tqdm(filenames):
                # print(filename)
                self.datapath.append(os.path.join(root, split, 'cloud', filename))
        self.labels = np.concatenate(self.labels)
        if balance_data:
            self.balance_data()

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        pc = o3d.io.read_point_cloud(self.datapath[index])
        feats = np.array(pc.points)
        # normals = np.array(pc.normals)
        coords = np.floor(feats / self.voxel_size)
        feats /= 0.15
        # feats = np.concatenate([feats, normals], axis=-1)
        return (coords, feats, self.labels[index])

    def balance_data(self):
        pos_datapath = np.array(self.datapath)[self.labels==1]
        neg_datapath = np.array(self.datapath)[self.labels==0]
        pos_labels = self.labels[self.labels==1]
        neg_labels = self.labels[self.labels==0]
        step = neg_labels.shape[0] / float(pos_labels.shape[0])
        neg_indices = np.unique([int(i*step) for i in range(pos_labels.shape[0])])
        neg_datapath = neg_datapath[neg_indices]
        neg_labels = neg_labels[neg_indices]
        self.datapath = np.concatenate([pos_datapath, neg_datapath]).tolist()
        self.labels = np.concatenate([pos_labels, neg_labels])
        print('pos: %d, neg: %d' % (pos_labels.shape[0], neg_labels.shape[0]))

def make_data_loader(dset, batch_size, shuffle, num_workers, repeat):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_pointcloud_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dset, shuffle)
        # args['sampler'] = RandomSampler(dset, replacement=True, num_samples=5000)
    else:
        args['shuffle'] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)
    return loader

def test(net, test_iter, config, phase='val', step=0):
    net.eval()
    num_correct, tot_num = 0, 0
    tp0,tp1,tp2 = 0,0,0
    positive0,positive1,positive2 = 0,0,0
    predict0,predict1,predict2 = 0,0,0
    num = 10
    for i in range(num):
        data_dict = test_iter.next()

        sin = ME.SparseTensor(
            data_dict['feats'],
            data_dict['coords'].int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        ).to(device)
        sout = net(sin)
        # print(len(sout))

        predict = torch.argmax(sout.F, 1).cpu()
        # print(predict)

        is_correct = data_dict['labels'] == predict
        num_correct += is_correct.sum().item()
        tot_num += len(sout)

        tp0 += ((data_dict['labels']==predict)&(data_dict['labels']==0)).sum().item()
        tp1 += ((data_dict['labels']==predict)&(data_dict['labels']==1)).sum().item()
        positive0 += (data_dict['labels']==0).sum().item()
        positive1 += (data_dict['labels']==1).sum().item()
        predict0 += (predict==0).sum().item()
        predict1 += (predict==1).sum().item()

    p0 = tp0/(predict0+1)
    p1 = tp1/(predict1+1)
    r0 = tp0/(positive0+1)
    r1 = tp1/(positive1+1)
    f0=2*p0*r0/(p0+r0+1)
    f1=2*p1*r1/(p1+r1+1)
    # test_logger.info(f'{phase} set accuracy : 
    log_string(test_logger, f'{phase} set accuracy : {num_correct / tot_num:.3e}, precision : { (p0+p1)/2 :.3e}, recall : {(r0+r1)/2 :.3e}, fscore : {(f0+f1)/2 :.3e}')
    log_string(test_logger, f'{phase} class0 precision : {p0:.3e}, recall : {r0:.3e}, fscore : {f0:.3e}')
    log_string(test_logger, f'{phase} class1 precision : {p1:.3e}, recall : {r1:.3e}, fscore : {f1:.3e}')
    test_writer.add_scalar('accuracy', num_correct / tot_num, step)
    test_writer.add_scalar('precision', (p0+p1)/2, step)
    test_writer.add_scalar('recall', (r0+r1)/2, step)
    test_writer.add_scalar('neg precision', p0, step)
    test_writer.add_scalar('neg recall', r0, step)
    test_writer.add_scalar('neg fscore', f0, step)
    test_writer.add_scalar('pos precision', p1, step)
    test_writer.add_scalar('pos recall', r1, step)
    test_writer.add_scalar('pos fscore', f1, step)
    return num_correct / tot_num

def train(net, device, config):
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    crit = torch.nn.CrossEntropyLoss()

    root = '/DATA2/chenxi/MinkowskiEngine/data/gpd_data_v4/'
    TRAIN_DATASET = GraspNetSVDataset(root, splits=['train1','train2','train3','train4'], voxel_size=config.voxel_size, balance_data=False)
    TEST_DATASET = GraspNetSVDataset(root, splits=['test_seen'], voxel_size=config.voxel_size, balance_data=False)

    train_dataloader = make_data_loader(
        TRAIN_DATASET,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        )
    val_dataloader = make_data_loader(
        TEST_DATASET,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        )

    curr_iter = 0
    if os.path.exists(os.path.join(config.log_dir, config.checkpoint_weights)):
        checkpoint = torch.load(os.path.join(config.log_dir, config.checkpoint_weights))
        net.load_state_dict(checkpoint['state_dict'])
        i = checkpoint['curr_iter']
        log_string(train_logger, f'Load Checkpoint Iter: {i}')
        if config.load_optimizer.lower() == 'true':
            curr_iter = checkpoint['curr_iter'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
    
    train_iter = iter(train_dataloader)
    val_iter = iter(val_dataloader)
    accuracy_max = test(net, val_iter, config, 'val', 0)
    net.train()
    log_string(train_logger, f'LR: {scheduler.get_last_lr()}')
    for i in range(curr_iter, config.max_iter):
        s = time()
        data_dict = train_iter.next()
        d = time() - s

        optimizer.zero_grad()
        sin = ME.SparseTensor(
            feats=data_dict['feats'],
            coords=data_dict['coords'].int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        ).to(device)
        sout = net(sin)

        labels = data_dict['labels'].to(device)
        loss = crit(sout.F, labels)
        loss.backward()
        optimizer.step()
        t = time() - s

        pred = torch.argmax(sout.F, 1)
        accuracy = pred.eq(labels.data).cpu().float().mean().item()
        train_writer.add_scalar('accuracy', accuracy, i)
        train_writer.add_scalar('loss', loss.item(), i)

        if i % config.stat_freq == 0:
            log_string(train_logger,
                f'Iter: {i}, Loss: {loss.item():.3e}, Data Loading Time: {d:.3e}, Tot Time: {t:.3e}'
            )

        if i % config.val_freq == 0 and i > 0:
            # Validation
            log_string(test_logger, 'Validation')
            accuracy = test(net, val_iter, config, 'val', i)
            if accuracy>accuracy_max:
                accuracy_max=accuracy
                torch.save(
                    {
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'curr_iter': i,
                    }, os.path.join(config.log_dir, config.weights))
            scheduler.step()
            log_string(train_logger, f'LR: {scheduler.get_last_lr()}')
            net.train()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = ResNet50(3, 11, D=3)
    # net = ResNet50(6, 2, D=3)
    # checkpoint = torch.load('modelnet.pth')
    # net.load_state_dict(checkpoint['state_dict'])
    net.to(device)

    train(net, device, config)

    # test_dataloader = make_data_loader(
    #     'test',
    #     augment_data=False,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers,
    #     repeat=False,
    #     config=config)

    # test_logger.info('Test')
    # test(net, iter(test_dataloader), config, 'test')
