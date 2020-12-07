import argparse
import os
import time
import pickle

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from model.dataset import *
from model.pointnet import PointNetCls, DualPointNetCls

parser = argparse.ArgumentParser(description='pointnetGPD')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
<<<<<<< Updated upstream
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--cuda', action='store_true')
<<<<<<< HEAD
parser.add_argument('--gpu', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.0005)
=======
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.005)
>>>>>>> a9fc64760b36adc06d7d772e52c04514aa96e708
=======
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.0005)
>>>>>>> Stashed changes
parser.add_argument('--load-model', type=str, default='')
parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                   help='pre-trained model path')
parser.add_argument('--data-path', type=str, default='/DATA2/chenxi/MinkowskiEngine/data/gpd_data_v4/', help='data path')
<<<<<<< Updated upstream
<<<<<<< HEAD
parser.add_argument('--log-interval', type=int, default=1000)
=======
parser.add_argument('--log-interval', type=int, default=10)
>>>>>>> a9fc64760b36adc06d7d772e52c04514aa96e708
=======
parser.add_argument('--log-interval', type=int, default=1000)
>>>>>>> Stashed changes
parser.add_argument('--save-interval', type=int, default=1)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)
    print("yes, we can run cuda")
logger = SummaryWriter(os.path.join('./assets/log/', args.tag))
np.random.seed(int(time.time()))

def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31-1))

def my_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

grasp_points_num=750
point_channel=3
# FCs for dataset
# fc_list_sub1 = np.arange(2.0, 0.75, -0.4)
# fc_list_sub2 = np.arange(0.5, 0.36, -0.05)
# ->[0.4, 0.5] class 1 (0.4, 0.45, 0.5)
# ->(0.5, 1.2] class 2 (0.8, 1.2)
# ->(1.2, 2.0] class 3 (1.6, 2.0)
thresh_bad=1.2
thresh_good=0.5

<<<<<<< Updated upstream
train_loader = torch.utils.data.DataLoader(
    PointGraspOneViewMultiClassDataset(
        grasp_points_num=grasp_points_num,
        path=args.data_path,
        tag='train',
        grasp_amount_per_file=6500,
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
    ),
    batch_size=args.batch_size,
    num_workers=32,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

test_loader = torch.utils.data.DataLoader(
    PointGraspOneViewMultiClassDataset(
        grasp_points_num=grasp_points_num,
        path=args.data_path,
        tag='test',
        grasp_amount_per_file=500,
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
        with_obj=True,
    ),
    batch_size=args.batch_size,
    num_workers=32,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)
=======
if args.mode == 'train':
    train_loader = torch.utils.data.DataLoader(
        PointGraspOneViewMultiClassDataset(
            grasp_points_num=grasp_points_num,
            path=args.data_path,
            tag='train',
            grasp_amount_per_file=6500,
            thresh_good=thresh_good,
            thresh_bad=thresh_bad,
        ),
        batch_size=args.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )
else:
    test_loader = torch.utils.data.DataLoader(
        PointGraspOneViewMultiClassDataset(
            grasp_points_num=grasp_points_num,
            path=args.data_path,
            tag='test',
            grasp_amount_per_file=500,
            thresh_good=thresh_good,
            thresh_bad=thresh_bad,
            with_obj=True,
        ),
        batch_size=args.batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate,
    )
>>>>>>> Stashed changes

is_resume = 0
if args.load_model and args.load_epoch != -1:
    is_resume = 1

if is_resume or args.mode == 'test':
    model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
    model.device_ids = [args.gpu]
    print('load model {}'.format(args.load_model))
else:
    model = PointNetCls(num_points=grasp_points_num, input_chann=point_channel, k=3)
if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0,1,2,3]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

def train(model, loader, epoch):
    print('we are training')
<<<<<<< Updated upstream
<<<<<<< HEAD
    optimizer.step()
=======
>>>>>>> a9fc64760b36adc06d7d772e52c04514aa96e708
=======
    optimizer.step()
>>>>>>> Stashed changes
    scheduler.step()
    model.train()
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    for batch_idx, (data, target) in enumerate(loader):
<<<<<<< Updated upstream
<<<<<<< HEAD
        #print("Now is the batch" + str(batch_idx))
=======
        print("Now is the batch" + str(batch_idx))
>>>>>>> a9fc64760b36adc06d7d772e52c04514aa96e708
=======
        #print("Now is the batch" + str(batch_idx))
>>>>>>> Stashed changes
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(
            epoch, batch_idx * args.batch_size, len(loader.dataset),
            100. * batch_idx * args.batch_size / len(loader.dataset), loss.item(), args.tag))
            logger.add_scalar('train_loss', loss.cpu().item(),
                    batch_idx + epoch * len(loader))
    return float(correct)/float(dataset_size)


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    da = {}
    db = {}
    res = []
<<<<<<< Updated upstream
<<<<<<< HEAD
    for data, target in loader:
=======
    for data, target, obj_name in loader:
>>>>>>> a9fc64760b36adc06d7d772e52c04514aa96e708
=======
    for data, target in loader:
>>>>>>> Stashed changes
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
<<<<<<< Updated upstream
        output, _ = model(data) # N*C
<<<<<<< HEAD
=======
        # data <class 'torch.Tensor'>
        output, _ = model(data) # N*C
>>>>>>> Stashed changes
        test_loss += F.nll_loss(output, target, reduction='sum').cpu().item()
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        '''
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, j[0], k))
        '''
<<<<<<< Updated upstream
=======
        test_loss += F.nll_loss(output, target, size_average=False).cpu().item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, j[0], k))

>>>>>>> a9fc64760b36adc06d7d772e52c04514aa96e708
=======
>>>>>>> Stashed changes
    test_loss /= len(loader.dataset)
    acc = float(correct)/float(dataset_size)
    return acc, test_loss


def main():
    if args.mode == 'train':
        for epoch in range(is_resume*args.load_epoch, args.epoch):
            acc_train = train(model, train_loader, epoch)
            print('Train done, acc={}'.format(acc_train))
            acc, loss = test(model, test_loader)
            print('Test done, acc={}, loss={}'.format(acc, loss))
            logger.add_scalar('train_acc', acc_train, epoch)
            logger.add_scalar('test_acc', acc, epoch)
            logger.add_scalar('test_loss', loss, epoch)
            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.model'.format(epoch))
                torch.save(model, path)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        acc, loss = test(model, test_loader)
        print('Test done, acc={}, loss={}'.format(acc, loss))

if __name__ == "__main__":
    main()
