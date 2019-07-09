#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import sys

# sys.getdefaultencoding()

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

from src.model import weight_init, LinearModel_Drover, Discriminator
from src.datasets.human36m import Human36M
import src.spherical_coords as spherical_coords
from tensorboardX import SummaryWriter



def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    data_path = 'D:\\Human pose DB\\baseline_data\\data'
    train_3d = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))  ## (48,)
    train_2d = torch.load(os.path.join(data_path, 'train_2d.pth.tar'))  ## (32, )

    test_3d = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))
    test_2d = torch.load(os.path.join(data_path, 'test_2d.pth.tar'))

    # summary = SummaryWriter()
    # save options
    log.save_options(opt, opt.ckpt)

    # create model
    print(">>> creating model")
    model = LinearModel_Drover()
    model = model.cuda()
    model.apply(weight_init)

    ## GAN discriminator model
    discriminator = Discriminator()
    discriminator = discriminator.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    simple_loss = nn.MSELoss(reduction='mean').cuda()
    simple_optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    ## GAN loss and optim
    gan_loss = nn.BCELoss(reduction='mean').cuda()
    gan_optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load, encoding='utf-8')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        simple_optimizer.load_state_dict(ckpt['optimizer'])
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
            print(">>> TEST on _{}_".format(action))

            test_loader = DataLoader(
                dataset=Human36M(actions=actions, data_path=opt.data_dir,
                                 train_2d=train_2d, train_3d=train_3d,
                                 test_2d=test_2d, test_3d=test_3d,
                                 set_num_samples=opt.set_num_samples,
                                 use_hg=opt.use_hg, is_train=False),
                batch_size=opt.test_batch,
                shuffle=False,
                num_workers=opt.job,
                pin_memory=True)

            _, err_test = test(test_loader, model, simple_loss, procrustes=opt.procrustes)
            err_set.append(err_test)

        print(">>>>>> TEST results:")

        for action in actions:
            print("{}".format(action), end='\t')
        print("\n")

        for err in err_set:
            print("{:.4f}".format(err), end='\t')
        print(">>>\nERRORS: {}".format(np.array(err_set).mean()))
        sys.exit()

    # load datasets for training
    test_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir,
                         train_2d=train_2d, train_3d=train_3d,
                         test_2d=test_2d, test_3d=test_3d,
                         set_num_samples=opt.set_num_samples,
                         use_hg=opt.use_hg, is_train=False),
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)

    train_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir,
                         train_2d=train_2d, train_3d=train_3d,
                         test_2d=test_2d, test_3d=test_3d,
                         set_num_samples=opt.set_num_samples,
                         use_hg=opt.use_hg),
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
        glob_step, lr_now, loss_train, loss_d = train(
            train_loader, model, simple_loss, simple_optimizer, gan_optimizer_D, gan_loss, discriminator,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        # test
        loss_test, err_test = test(test_loader, model, simple_loss, procrustes=opt.procrustes)

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
                           'optimizer': simple_optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=True)

        else:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': simple_optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=False)

    logger.close()

def random_rotation(X):
    # random projection
    az_angle = np.random.uniform(-math.pi, math.pi, X.size(0))
    el_angle = np.random.uniform(-math.pi / 9, math.pi / 9, X.size(0))
    R = torch.zeros(X.size(0), 3, 3)
    R_inv = torch.zeros(X.size(0), 3, 3)

    for k in range(X.size(0)):
        [Theta, Phi] = spherical_coords.azel_to_thetaphi(az_angle[k], el_angle[k])
        # Theta = math.acos(math.cos(el_angle[k]) * math.cos(az_angle[k]))
        # Phi = math.atan(math.tan(el_angle[k]) / math.sin(az_angle[k]))
        # Theta = el_angle[k]
        # Phi = az_angle[k]
        Rx = torch.zeros([3, 3])
        Rx[0, 0] = 1
        Rx[1, 1] = math.cos(Phi)
        Rx[1, 2] = -math.sin(Phi)
        Rx[2, 1] = math.sin(Phi)
        Rx[2, 2] = math.cos(Phi)

        Ry = torch.zeros([3, 3])
        Ry[1, 1] = 1
        Ry[0, 0] = math.cos(Theta)
        Ry[0, 2] = math.sin(Theta)
        Ry[2, 0] = -math.sin(Theta)
        Ry[2, 2] = math.cos(Theta)

        R[k, :, :] = torch.matmul(Ry, Rx)
        R_inv[k, :, :] = torch.inverse(R[k, :, :])

    X_reshape = X.reshape((-1, 16, 3)).permute((0, 2, 1))  # N by 3 by num_joint
    Xr = X_reshape[:, :, 0].reshape((-1, 3, 1))
    T = torch.zeros((X_reshape.size(0), X_reshape.size(1), X_reshape.size(2)))
    T[:, 2, :] = 10

    Y = torch.matmul(R.cuda(), X_reshape - Xr) + T.cuda()
    Y_reshape = Y.permute((0, 2, 1)).reshape(-1, 16 * 3)  # N by num_joint * 3
    y = (Y[:, :2, :] / Y[:, 2, :].reshape(-1, 1, 16)).permute((0, 2, 1)).reshape((-1, 16 * 2))

    return y, Y_reshape, R_inv, T, Xr

def inverse_projection(Y_tilde, R_inv, T, Xr):
    # inverse projection
    Y_tilde_reshape = Y_tilde.reshape((-1, 16, 3)).permute((0, 2, 1))  # N by 3 by num_joint
    X_tilde = torch.matmul(R_inv.cuda(), Y_tilde_reshape - T.cuda()) + Xr
    x_tilde = (X_tilde[:, :2, :] / X_tilde[:, 2, :].reshape(-1, 1, 16)).permute((0, 2, 1)).reshape((-1, 16 * 2))

    return x_tilde

def train(train_loader, model, simple_loss, simple_optimizer, gan_optimizer_D, gan_loss, discriminator,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True):

    losses = utils.AverageMeter()
    D_losses = utils.AverageMeter()

    model.train()
    model_eval = model.eval()

    discriminator.train()
    gan_eval = discriminator.eval()

    pbar = tqdm(train_loader)
    for i, (inps, tars) in enumerate(pbar):  # inps = (64, 32)
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(simple_optimizer, glob_step, lr_init, lr_decay, gamma)

        inputs_use = inps.data.cpu().numpy()
        inputs_dist_norm, _ = data_process.input_norm(inputs_use, d_ref=10)  # (64, 32) , array
        input_dist = torch.tensor(inputs_dist_norm, dtype=torch.float32)

        targets_use = tars.data.cpu().numpy()
        targets_dist_norm, _ = data_process.output_norm(targets_use)
        targets_dist = torch.tensor(targets_dist_norm, dtype=torch.float32)

        inputs = Variable(input_dist.cuda())
        targets = Variable(targets_dist.cuda(async=True))

        X = model(inputs)  # prediction 3d pose [batchsize, num joint * 3]

        y, Y_reshape, R_inv, T, Xr = random_rotation(X)

        Y_tilde = model_eval(y)  # prediction 3d pose [batchsize, num joint * 3]
        x_tilde = inverse_projection(Y_tilde, R_inv, T, Xr)

        ### GAN
        ## make label
        Tensor = torch.cuda.FloatTensor
        valid = Variable(Tensor(inps.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(inps.size(0), 1).fill_(0.0), requires_grad=False)
        real_lb = torch.cat((valid, fake), 1)
        fake_lb = torch.cat((fake, valid), 1)

        #### Generator loss
        simple_optimizer.zero_grad()
        loss_g = gan_loss(discriminator(y), real_lb)  # .detach()
        loss2d = simple_loss(inputs, x_tilde)
        loss3d = simple_loss(Y_reshape, Y_tilde)
        loss = (0.001 * loss3d) + (10 * loss2d) + loss_g
        # loss = loss3d + loss2d + loss_g


        loss.backward()
        loss_temp = simple_loss(targets, X)
        losses.update(loss_temp.item(), inputs.size(0))

        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        simple_optimizer.step()

        ##### Discriminator loss
        gan_optimizer_D.zero_grad()
        real_loss = gan_loss(discriminator(inputs), real_lb)
        fake_loss = gan_loss(discriminator(y.detach()), fake_lb)
        loss_d = (real_loss + fake_loss) / 2

        loss_d.backward()
        D_losses.update(loss_d.item(), inputs.size(0))
        gan_optimizer_D.step()

        # pbar.set_postfix(tr_loss='{:05.6f}'.format(losses.avg),  loss_2d='{:05.6f}'.format(loss2d.item()),
        #                  loss_3d='{:05.6f}'.format(loss3d.item()), loss_d='{:05.6f}'.format(loss_d.item()))

        pbar.set_postfix(G_loss='{:05.6f}'.format(losses.avg), D_loss='{:05.6f}'.format(D_losses.avg))


    return glob_step, lr_now, losses.avg, D_losses.avg



def test(test_loader, model, simple_loss, procrustes=False):
    losses = utils.AverageMeter()


    model.eval()

    all_dist = []
    pbar = tqdm(test_loader)
    for i, (inps, tars) in enumerate(pbar):

        ### input dist norm
        data_use = inps.data.cpu().numpy()
        data_dist_norm, data_dist_set = data_process.input_norm(data_use, d_ref=10)  # (64, 32) , array
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
        loss = simple_loss(pred_coord, targets)  # 64 losses average

        losses.update(loss.item(), inputs.size(0))

        tars = targets
        pred = outputs

        targets_dist = np.reshape(label_dist_set, (-1, 1))
        targets_dist_set = np.repeat(targets_dist, 48, axis=1)

        c = np.reshape(np.asarray([0, 0, 10]), (1, -1))
        c = np.repeat(c, 16, axis=0)
        c = np.reshape(c, (1, -1))
        c = np.repeat(c, inputs.size(0), axis=0)

        #### undist -> unnorm
        outputs_undist = (pred.data.cpu().numpy() * targets_dist_set) - c
        targets_undist = (tars.data.cpu().numpy() * targets_dist_set) - c

        outputs_use = outputs_undist
        targets_use = targets_undist  # (64, 48)

        if procrustes:
            for ba in range(inps.size(0)):
                gt = targets_use[ba].reshape(-1, 3)
                out = outputs_use[ba].reshape(-1, 3)  # (17,3)
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
    print(">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err


if __name__ == "__main__":
    option = Options().parse()
    option.set_num_samples = -1
    option.procrustes = False
    option.test = False
    option.resume = False  # If you want to resume train from previous ckpt then set as True. Also, have to set option.load file path
    # option.load = 'D:\\Workspace\\3d_pose_baseline_pytorch-master\\3d_pose_baseline_pytorch-master\\checkpoint\\test\\ckpt_best.pth.tar' # file_path where ckpt files are in
    option.load = ''  # file_path where ckpt files are in
    main(option)
    # print(main)


