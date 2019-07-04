from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from src.procrustes import get_transformation


import src.data_process as data_process
import src.utils as utils



def test(test_loader, model, criterion, procrustes=False):

    losses = utils.AverageMeter()

    model.eval()

    all_dist = []
    # start = time.time()
    # batch_time = 0
    # bar = Bar('>>>', fill='>', max=len(test_loader))

    # for i, (inps, tars) in enumerate(test_loader):

    pbar = tqdm(test_loader)
    for i, (inps, tars) in enumerate(pbar):

        ### input unnorm
        # data_coord = data_process.unNormalizeData(inps.data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use']) # 64, 64
        # dim_2d_use = stat_2d['dim_use']
        # data_use = data_coord[:, dim_2d_use]  # (64, 32)

        ### input dist norm
        data_dist_norm, data_dist_set = data_process.input_norm(inps.data.cpu().numpy()) # (64, 32) , array
        data_dist = torch.tensor(data_dist_norm, dtype=torch.float32)

        # target unnorm
        # label_coord = data_process.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use']) # (64, 96)
        # dim_3d_use = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22,
        #                        23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58,
        #                        59, 75, 76, 77, 78, 79, 80, 81, 82, 83])

        # label_use = label_coord[:, dim_3d_use]  # (48, )
        # target dist norm
        label_dist_norm, label_dist_set = data_process.output_norm(tars.data.cpu().numpy())
        label_dist = torch.tensor(label_dist_norm, dtype=torch.float32)

        inputs = Variable(data_dist.cuda())
        targets = Variable(label_dist.cuda(async=True))

        outputs = model(inputs)

        # calculate loss
        pred_coord = outputs
        loss = criterion(pred_coord, targets) # 64 losses average

        losses.update(loss.item(), inputs.size(0))

        # tars = targets
        # pred = outputs

        # inputs_dist_set = np.reshape(targets_dist_set, (-1, 1))
        # inputs_dist_set = np.repeat(targets_dist_set, 48, axis=1)

        targets_dist = np.reshape(label_dist_set, (-1, 1))
        targets_dist_set = np.repeat(targets_dist, 48, axis=1)

        c = np.reshape( np.asarray( [0,0,10]) , (1,-1) )
        c = np.repeat(c, 16, axis=0)
        c = np.reshape(c, (1, -1) )
        c = np.repeat(c, inputs.size(0) , axis=0)
        # c_set = np.repeat(np.asarray([0,0,10]), 16, axis=0)

        #### undist -> unnorm
        # outputs_undist = (outputs.data.cpu().numpy() * targets_dist_set) -c
        outputs_undist = (outputs.data.cpu().numpy() * targets_dist_set) - c

        # outputs_undist = outputs_undist - c
        # targets_undist =  ( targets.data.cpu().numpy() * targets_dist_set ) - c
        targets_undist =  ( targets.data.cpu().numpy() * targets_dist_set ) -c

        # targets_undist = targets_undist - c


        #####
        '''targets_unnorm : 
         -8095.789092890324,
 3447.739184577949,
 -12287.197684264487,
 29496.41833447592,
 91899.94409139897,
 55478.075331596345,
 -26361.992715253342,
        '''
        # targets_unnorm = data_process.unNormalizeData(outputs_undist, stat_3d['mean'], stat_3d['std'],
        #                                               dim_3d_use)
        # outputs_unnorm = data_process.unNormalizeData(targets_undist, stat_3d['mean'], stat_3d['std'],
        #                                               dim_3d_use)


        # inputs_unnorm = data_process.unNormalizeData(inputs.data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])
        # calculate erruracy

        # targets_unnorm = data_process.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        # outputs_unnorm = data_process.unNormalizeData(outputs.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        # remove dim ignored

        # dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))
        # dim_use = dim_3d_use


        # dim_2d_use = stat_2d['dim_use']

        # stat_3d['dim_use'] (48., )
        # outputs_use = outputs_unnorm[:, dim_use]
        # targets_use = targets_unnorm[:, dim_use] # (64, 48)

        outputs_use = outputs_undist
        targets_use = targets_undist# (64, 48)

        # if i == 300:
        #     plot_3d(outputs_use, targets_use,  20)
        #     break

        # inputs_use = inputs_unnorm[:, dim_2d_use] # (64, 32)

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
        # for k in np.arange(0, 17 * 3, 3):

            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            dist_idx += 1
        all_dist.append(distance)

        # # update summary
        # if (i + 1) % 100 == 0:
        #     batch_time = time.time() - start
        #     start = time.time()
        #
        # bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
        #     .format(batch=i + 1,
        #             size=len(test_loader),
        #             batchtime=batch_time * 10.0,
        #             ttl=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             loss=losses.avg)
        # bar.next()

        pbar.set_postfix(tt_loss='{:05.6f}'.format(losses.avg))

        # ### visualization
        # if i == 500:
        #     simple_3d_line(outputs_use, 10)
        #     simple_3d_line(targets_use, 10)
        #     break


    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)
    # bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err