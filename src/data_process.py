#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import math
import torch



def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_use):
    T = normalized_data.shape[0]  # Batch size
    D = data_mean.shape[0]  # 96

    orig_data = np.zeros((T, D), dtype=np.float32) # (64, 96)
    orig_data[:, dimensions_to_use] = normalized_data # (64, 48)

    # Multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D)) # (1, 96)
    stdMat = np.repeat(stdMat, T, axis=0) # (64 , 96)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data


### Input unnormalization for Z Score
# inputs_unnorm = data_process.unNormalizeData(inps.data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use']) # 64, 64
# unnorm size = 64, make zeros mtrx and just do unnorm, so (0 * stdMat) + meanMat => 64, 64 // junk values the other position except original 16 joints
# dim_2d_use = stat_2d['dim_use']
# select useful 32 joint using dim_2d-use  => 64, 32
# inputs_use = inputs_unnorm[:, dim_2d_use]  # (64, 32)

### Targets unnormalization
# targets_unnorm = data_process.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use']) # (64, 96)
# dim_3d_use = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22,
#                        23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58,
#                        59, 75, 76, 77, 78, 79, 80, 81, 82, 83])
# targets_use = targets_unnorm[:, dim_3d_use] # (51, )

def unNormalize2dData(normalized_data, data_mean, data_std):

    dimensions_to_use = np.asarray([1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27])

    T = normalized_data.shape[0]

    cut_data_mean = data_mean.reshape(-1, 2)
    orig_data_mean = cut_data_mean[dimensions_to_use, :]
    orig_data_mean = orig_data_mean.reshape(1, 32)
    meanMat =  np.repeat(orig_data_mean, T, axis=0)

    cut_data_std = data_std.reshape(-1, 2)
    orig_data_std = cut_data_std[dimensions_to_use, :]
    orig_data_std = orig_data_std.reshape(1, 32)
    stdMat =  np.repeat(orig_data_std, T, axis=0)

    orig_data = np.multiply(np.asarray(normalized_data), stdMat) + meanMat

    return orig_data

def unNormalize3dData(normalized_data, data_mean, data_std):

    dimensions_to_use = np.asarray([1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27])

    T = normalized_data.shape[0]

    cut_data_mean = data_mean.reshape(-1, 3)
    orig_data_mean = cut_data_mean[dimensions_to_use, :]
    orig_data_mean = orig_data_mean.reshape(1, 48)
    meanMat =  np.repeat(orig_data_mean, T, axis=0)

    cut_data_std = data_std.reshape(-1, 3)
    orig_data_std = cut_data_std[dimensions_to_use, :]
    orig_data_std = orig_data_std.reshape(1, 48)
    stdMat =  np.repeat(orig_data_std, T, axis=0)

    orig_data = np.multiply(np.asarray(normalized_data), stdMat) + meanMat

    return orig_data


def input_norm(inputs, d_ref):

    head_idx = 9
    hip_idx = 0
    # r_hip_idx = 1
    # l_hip_idx = 4
    input_set = []
    input_size = len(inputs)
    dist_set = []

    for idx in range(len(inputs)):
        # inputs = np.asarray(inputs)
        current_sample = inputs[idx]
        current_sample = current_sample.reshape(16,2)
        # central_point = (current_sample[r_hip_idx-1] + current_sample[l_hip_idx-1]) / 2
        current_sample = current_sample - current_sample[hip_idx] # make root [0, 0]
        central_point = current_sample[hip_idx]
        head_point = current_sample[head_idx]
        dist = math.sqrt(np.sum((central_point - head_point) ** 2))

        new_sample = current_sample / (d_ref * dist)
        new_sample = np.reshape(new_sample, (1, 32))
        input_set.append(new_sample)
        dist_set.append(dist)

    input_set = np.asarray(input_set).reshape(input_size, 32)
    dist_set = np.asarray(dist_set)

    return input_set, dist_set

        ##### check head, hip, central point
        # r_hip = current_sample[r_hip_idx - 1]
        # l_hip = current_sample[l_hip_idx - 1]
        # head = current_sample[head_idx - 1]
        # coord = np.asarray([np.asarray(head), np.asarray(l_hip), np.asarray(r_hip), np.asarray(central_point)])
        # x_coord = coord[:,0]
        # y_coord = coord[:,1]
        # plt.figure()
        # plt.scatter(x_coord, y_coord)


def output_norm(outputs):

    head_idx = 9
    hip_idx = 0
    # r_hip_idx = 1
    # l_hip_idx = 4
    c = [0,0,10]
    output_set = []
    dist_set = []
    output_size = len(outputs)

    for idx in range(len(outputs)):
        current_sample = outputs[idx]
        current_sample = current_sample.reshape(16, 3)
        adjust_current_sample = current_sample

        # central_point = (current_sample[r_hip_idx - 1] + current_sample[l_hip_idx - 1]) / 2
        # central_point = torch.tensor([0, 0, c], dtype=torch.float32)
        head_point = adjust_current_sample[head_idx]
        central_point = adjust_current_sample[hip_idx]

        # head_point = torch.tensor(current_sample[head_idx], dtype=torch.float32)
        # distance = (central_point - head_point) ** 2
        # dist = math.sqrt(distance.sum())
        dist = math.sqrt(np.sum((central_point - head_point) ** 2))


        new_sample = (adjust_current_sample / dist) + c
        new_sample = np.reshape(new_sample, (1, 48))
        output_set.append(new_sample)
        dist_set.append(dist)


    output_set = np.asarray(output_set).reshape(output_size, 48)
    dist_set = np.asarray(dist_set)

    return output_set, dist_set






