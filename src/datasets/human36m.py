#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
import numpy as np
from torch.utils.data import Dataset

#
# data_path = 'D:\\Human pose DB\\baseline_data\\data'
#
# train_3d = torch.load(os.path.join(data_path, 'train_3d.pth.tar'))  ## (48,)
# train_2d = torch.load(os.path.join(data_path, 'train_2d.pth.tar'))  ## (32, )
#
# test_3d = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))
# test_2d = torch.load(os.path.join(data_path, 'test_2d.pth.tar'))


class Human36M(Dataset):
    def __init__(self, actions, data_path, set_num_samples, train_2d, train_3d, test_2d, test_3d, use_hg=True, is_train=True):
        """
        :param actions: list of actions to use
        :param data_path: path to dataset
        :param use_hg: use stacked hourglass detections
        :param is_train: load train/test dataset
        """

        self.actions = actions
        self.data_path = data_path
        self.set_num_samples = set_num_samples

        self.is_train = is_train
        self.use_hg = use_hg

        self.train_inp, self.train_out, self.test_inp, self.test_out = [], [], [], []
        self.train_meta, self.test_meta = [], []

        self.dim_2d_use = [ 0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15, 16, 17, 24, 25, 26,
       27, 30, 31, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 54, 55]

        self.dim_3d_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22,
                               23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58,
                               59, 75, 76, 77, 78, 79, 80, 81, 82, 83]



        # loading data
        if self.use_hg:
            train_2d_file = 'train_2d_ft.pth.tar'
            test_2d_file = 'test_2d_ft.pth.tar'
        else:
            train_2d_file = 'train_2d.pth.tar'
            test_2d_file = 'test_2d.pth.tar'
            # stat_2d_file = 'stat_2d.pth.tar'

        if self.is_train:
            # load train data
            # self.train_3d = torch.load(os.path.join(data_path, 'train_3d.pth.tar')) ## (48,)
            # self.train_2d = torch.load(os.path.join(data_path, train_2d_file)) ## (32, )
            self.train_3d = train_3d
            self.train_2d = train_2d

            # self.stat_2d = torch.load(os.path.join(data_path, stat_2d_file))

            # data_std = self.stat_2d['std']
            # data_mean = self.stat_2d['mean']
            #

            for k2d in self.train_2d.keys():

                (sub, act, fname) = k2d
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d

                if not self.set_num_samples == -1:
                    num_f = self.set_num_samples
                else:
                    num_f, _ = self.train_2d[k2d].shape



                assert self.train_3d[k3d].shape[0] == self.train_2d[k2d].shape[0], '(training) 3d & 2d shape not matched'

                for i in range(num_f):
                    self.train_inp.append(self.train_2d[k2d][i])
                    self.train_out.append(self.train_3d[k3d][i])


        else:
            # load test data
            self.test_3d =test_3d
            self.test_2d =test_2d

            for k2d in self.test_2d.keys():
                (sub, act, fname) = k2d
                if act not in self.actions:
                    continue
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d

                if not self.set_num_samples == -1:
                    num_f = self.set_num_samples
                else:
                    num_f, _ = self.test_2d[k2d].shape

                # num_f, _ = self.test_2d[k2d].shape
                assert self.test_2d[k2d].shape[0] == self.test_3d[k3d].shape[0], '(test) 3d & 2d shape not matched'
                for i in range(num_f):
                    self.test_inp.append(self.test_2d[k2d][i])
                    self.test_out.append(self.test_3d[k3d][i])

    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            outputs = torch.from_numpy(self.train_out[index]).float()

        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()

        return inputs, outputs

    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)





#
# from opt import Options
# import src.misc as misc
#
#
# option = Options().parse()
#
# actions = misc.define_actions(option.action)
#
# Human36M(actions=actions[0], data_path=option.data_dir, use_hg=option.use_hg, is_train=False)
#
# print(actions)
