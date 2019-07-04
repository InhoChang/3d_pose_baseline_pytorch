#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from opt import Options
import src.misc as misc
import src.log as log

from src.model import LinearModel, weight_init, LinearModel_Drover, Discriminator
from src.datasets.human36m import Human36M

from src.test import test
from src.train import train


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    ## save options
    log.save_options(opt, opt.ckpt)

    ## create model
    print(">>> creating model")
    model = LinearModel()
    # model = LinearModel_Drover()
    model = model.cuda()
    model.apply(weight_init)

    ## GAN discriminator model
    discriminator = Discriminator()
    discriminator = discriminator.cuda()

    ## baseline loss and optim
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    ## GAN loss and optim
    gan_loss =  nn.BCELoss().cuda()
    gan_optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    ## load ckpt
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

    ## list of action(s)
    actions = misc.define_actions(opt.action)

    '''
    
    actions = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", 
    "Photo", "Posing", "Purchases", "Sitting", "SittingDown", "Smoking", 
    "Waiting", "WalkDog", "Walking", "WalkTogether"]
    
    '''

    ## Test Session
    # actions = ["Directions"]
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

    ## load datasets for training
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

    ## Train Session
    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        # train
        glob_step, lr_now, loss_train, loss_d= train(
            train_loader, model, criterion, optimizer, gan_optimizer_D, gan_loss, discriminator,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)

        # test
        loss_test, err_test = test(test_loader, model, criterion, procrustes=opt.procrustes)

        # update log file
        logger.append([epoch + 1, lr_now, loss_train, loss_test, err_test],
                      ['int', 'float', 'float', 'float', 'float'])

        # save best or last ckpt
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


if __name__ == "__main__":
    option = Options().parse()
    option.set_num_samples = 10
    option.procrustes = True
    option.test = False
    option.resume = False
    option.load = ''
    # option.load = 'D:\\Workspace\\3d_pose_baseline_pytorch-master\\3d_pose_baseline_pytorch-master\\checkpoint\\test\\ckpt_best.pth.tar' # file_path where ckpt files are in

    main(option)
    # print(main)


