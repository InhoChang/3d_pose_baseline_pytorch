from __future__ import print_function, absolute_import, division

from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

import src.data_process as data_process
import src.utils as utils



def train(train_loader, model, criterion, optimizer, gan_optimizer_D, gan_loss, discriminator,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True ):

    losses = utils.AverageMeter()
    D_losses = utils.AverageMeter()

    model.train()

    pbar = tqdm(train_loader)
    for i, (inps, tars) in enumerate(pbar): # inps = (64, 32)
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)

        inputs_dist_norm, _ = data_process.input_norm(inps.data.cpu().numpy()) # (64, 32) , array
        inputs_dist = torch.tensor(inputs_dist_norm, dtype=torch.float32)

        targets_dist_norm, _  = data_process.output_norm(tars.data.cpu().numpy())
        targets_dist = torch.tensor(targets_dist_norm, dtype=torch.float32)

        inputs = Variable(inputs_dist.cuda())
        targets = Variable(targets_dist.cuda(async=True))

        ## baseline output
        outputs = model(inputs)

        # Baseline calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        loss.backward()

        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        ### GAN
        ## make GAN label
        Tensor = torch.cuda.FloatTensor
        valid = Variable(Tensor(inps.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(inps.size(0), 1).fill_(0.0), requires_grad=False)

        real_lb = torch.cat( (valid, fake), 1)
        fake_lb = torch.cat( (fake, valid), 1)


        ## Warning !!!! This fake_joint is temporary input setting, after finish random rotation, inputs should be change to rotation_inputs
        fake_joint = torch.randn(inps.size(0), 32).cuda()   ## inputs => rotation_inputs

        ## GAN train
        gan_optimizer_D.zero_grad()
        real_loss = gan_loss(discriminator(inputs), real_lb)
        fake_loss = gan_loss(discriminator(fake_joint), fake_lb)
        # .detach()
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        D_losses.update(d_loss.item(), inputs.size(0))

        gan_optimizer_D.step()


        total_loss = loss + d_loss


        pbar.set_postfix(tr_loss='{:05.6f}'.format(losses.avg), d_loss='{:05.6f}'.format(D_losses.avg))


    return glob_step, lr_now, losses.avg, D_losses.avg