#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import sys
sys.getdefaultencoding()

import os
# import sys
from tqdm import tqdm

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from opt import Options
from src.procrustes import get_transformation
import src.data_process as data_process
import src.utils as utils
import src.misc as misc
import src.log as log

from src.model import weight_init, LinearModel_Drover
from src.datasets.human36m import Human36M


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)

    # create model
    print(">>> creating model")
    model = LinearModel_Drover()
    model = model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load, encoding='utf-8')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])

    # list of action(s)
    actions = misc.define_actions(opt.action)
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))

    # data loading
    print(">>> loading data")

    if opt.test:
        err_set = []
        for action in actions:
            print (">>> TEST on _{}_".format(action))

            test_loader = DataLoader(
                dataset=Human36M(actions=action, data_path=opt.data_dir,set_num_samples = opt.set_num_samples, use_hg=opt.use_hg, is_train=False),
                batch_size=opt.test_batch,
                shuffle=False,
                num_workers=opt.job,
                pin_memory=True)

            _, err_test = test(test_loader, model, criterion, procrustes=opt.procrustes)
            err_set.append(err_test)

        print (">>>>>> TEST results:")

        for action in actions:
            print ("{}".format(action), end='\t')
        print ("\n")

        for err in err_set:
            print ("{:.4f}".format(err), end='\t')
        print (">>>\nERRORS: {}".format(np.array(err_set).mean()))
        sys.exit()

    # load datasets for training
    test_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, set_num_samples = opt.set_num_samples, use_hg=opt.use_hg, is_train=False),
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)

    train_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, set_num_samples = opt.set_num_samples, use_hg=opt.use_hg),
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)

    print(">>> data loaded !")

    cudnn.benchmark = True

    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        ## per epoch
        # train
        glob_step, lr_now, loss_train = train(
            train_loader, model, criterion, optimizer,  lr_init=opt.lr, lr_now=lr_now,
            glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        # test
        loss_test, err_test = test(test_loader, model, criterion, procrustes=opt.procrustes)

        # update log file
        logger.append([epoch + 1, lr_now, loss_train, loss_test, err_test],
                      ['int', 'float', 'float', 'float', 'float'])

        # save ckpt
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        if is_best:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=True)

        else:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=False)

    logger.close()


def train(train_loader, model, criterion, optimizer, lr_init=None, lr_now=None, glob_step=None, lr_decay=None,
          gamma=None, max_norm=True ):

    losses = utils.AverageMeter()

    model.train()

    pbar = tqdm(train_loader)
    for i, (inps, tars) in enumerate(pbar): # inps = (64, 32)
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)

        inputs_use = inps.data.cpu().numpy()
        inputs_dist_norm, _ = data_process.input_norm(inputs_use, d_ref=10) # (64, 32) , array
        input_dist = torch.tensor(inputs_dist_norm, dtype=torch.float32)

        targets_use = tars.data.cpu().numpy()
        targets_dist_norm, _  = data_process.output_norm(targets_use)
        targets_dist = torch.tensor(targets_dist_norm, dtype=torch.float32)

        inputs = Variable(input_dist.cuda())
        targets = Variable(targets_dist.cuda(async=True))

        pred3d = model(inputs) # prediction 3d pose [batchsize, num joint * 3]

        # random projection
        random_azimu_angle = np.random.uniform(-math.pi, math.pi, pred3d.size(0))
        random_elev_angle = np.random.uniform(-math.pi/9, math.pi/9, pred3d.size(0))
        random_rot = torch.zeros(pred3d.size(0), 3, 3)
        for k in range(pred3d.size(0)):
            Az = torch.zeros([3, 3])
            Az[0, 0] = 1
            Az[1, 1] = math.cos(random_azimu_angle[k])
            Az[1, 2] = -math.sin(random_azimu_angle[k])
            Az[2, 1] = math.sin(random_azimu_angle[k])
            Az[2, 2] = math.cos(random_azimu_angle[k])

            El = torch.zeros([3, 3])
            El[1, 1] = 1
            El[0, 0] = math.cos(random_elev_angle[k])
            El[0, 2] = math.sin(random_elev_angle[k])
            El[2, 0] = -math.sin(random_elev_angle[k])
            El[2, 2] = math.cos(random_elev_angle[k])

            random_rot[k, :, :] = np.multiply(Az, El)

        pred3d_reshape = pred3d.reshape((-1, 16, 3))
        pred3d_origin = pred3d_reshape - pred3d_reshape[:, 0, :].reshape((-1, 1, 3))
        ref_translation = torch.zeros((pred3d_origin.size(0), pred3d_origin.size(1), pred3d_origin.size(2)))
        ref_translation[:,:,2] = 10

        pred3d_transform = torch.matmul(random_rot.cuda(), pred3d_origin.permute((0, 2, 1))).permute((0,2,1)) + ref_translation.cuda()
        pred3d_transform_reshape = pred3d_transform.reshape(-1, 16 * 3)

        reprojection_joint = pred3d_transform[:, :, :2] / pred3d_transform[:, :, 2].reshape(-1, 16, 1)
        inputs_repro = reprojection_joint.reshape((-1, 16 * 2))
        pred3d_random = model(inputs_repro) # prediction 3d pose [batchsize, num joint * 3]

        # inverse projection
        pred3d_random_reshape = pred3d_random.reshape((-1, 16, 3))
        pred3d_random_origin = pred3d_random_reshape - pred3d_random_reshape[:, 0, :].reshape((-1, 1, 3))
        random_rot_transpose = random_rot.permute((0, 2, 1))
        pred3d_random_invtransform = torch.matmul(random_rot_transpose.cuda(), pred3d_random_origin.permute((0, 2, 1))).permute(
            (0, 2, 1)) + ref_translation.cuda()
        random_reprojection_joint = pred3d_random_invtransform[:, :, :2] / pred3d_random_invtransform[:, :, 2].reshape(-1, 16, 1)
        random_reprojection_joint_final = random_reprojection_joint.reshape((-1, 16 * 2))

        # calculate loss
        optimizer.zero_grad()

        loss2d = criterion(inputs, random_reprojection_joint_final)
        loss3d = criterion(pred3d_transform_reshape, pred3d_random)

        loss = loss3d + loss2d

        losses.update(loss.item(), inputs.size(0))
        loss.backward()

        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        pbar.set_postfix(tr_loss='{:05.6f}'.format(losses.avg))

    return glob_step, lr_now, losses.avg

# def test(test_loader, model, criterion, stat2d, stat_3d, procrustes=False):

def test(test_loader, model, criterion, procrustes=False):

    losses = utils.AverageMeter()

    model.eval()

    all_dist = []
    pbar = tqdm(test_loader)
    for i, (inps, tars) in enumerate(pbar):

        ### input dist norm
        data_use = inps.data.cpu().numpy()
        data_dist_norm, data_dist_set = data_process.input_norm(data_use, d_ref=10) # (64, 32) , array
        data_dist = torch.tensor(data_dist_norm, dtype=torch.float32)

        # target dist norm
        label_use = tars.data.cpu().numpy()
        label_dist_norm, label_dist_set = data_process.output_norm(label_use)
        label_dist = torch.tensor(label_dist_norm, dtype=torch.float32)

        inputs = Variable(data_dist.cuda())
        targets = Variable(label_dist.cuda(async=True))

        outputs = model(inputs)

        # calculate loss
        pred_coord = outputs
        loss = criterion(pred_coord, targets) # 64 losses average

        losses.update(loss.item(), inputs.size(0))

        tars = targets
        pred = outputs

        targets_dist = np.reshape(label_dist_set, (-1, 1))
        targets_dist_set = np.repeat(targets_dist, 48, axis=1)

        c = np.reshape( np.asarray( [0,0,10]) , (1,-1) )
        c = np.repeat(c, 16, axis=0)
        c = np.reshape(c, (1, -1) )
        c = np.repeat(c, inputs.size(0) , axis=0)

        #### undist -> unnorm
        outputs_undist = (pred.data.cpu().numpy() * targets_dist_set) - c
        targets_undist =  (tars.data.cpu().numpy() * targets_dist_set ) - c

        outputs_use = outputs_undist
        targets_use = targets_undist# (64, 48)

        if procrustes:
            for ba in range(inps.size(0)):
                gt = targets_use[ba].reshape(-1, 3)
                out = outputs_use[ba].reshape(-1, 3) # (17,3)
                _, Z, T, b, c = get_transformation(gt, out, True)
                out = (b * out.dot(T)) + c
                outputs_use[ba, :] = out.reshape(1, 48)

        sqerr = (outputs_use - targets_use) ** 2

        # distance = np.zeros((sqerr.shape[0], 17))
        distance = np.zeros((sqerr.shape[0], 16))

        dist_idx = 0
        for k in np.arange(0, 16 * 3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            dist_idx += 1

        all_dist.append(distance)

        pbar.set_postfix(tt_loss='{:05.6f}'.format(losses.avg))

    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)
    # bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err


if __name__ == "__main__":
    option = Options().parse()
    option.set_num_samples = -1
    option.procrustes = False
    option.test = False
    option.resume = False # If you want to resume train from previous ckpt then set as True. Also, have to set option.load file path
    # option.load = 'D:\\Workspace\\3d_pose_baseline_pytorch-master\\3d_pose_baseline_pytorch-master\\checkpoint\\test\\ckpt_best.pth.tar' # file_path where ckpt files are in
    option.load = '' # file_path where ckpt files are in
    main(option)
    # print(main)


