#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import numpy as np
import os
import torch
from torch.utils.data import Dataset


TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

DIM_USE2 = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15, 16, 17, 24, 25, 26,
       27, 30, 31, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 54, 55],
      dtype=np.int64)

DIM_USE3 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22,
                       23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58,
                       59, 75, 76, 77, 78, 79, 80, 81, 82, 83],
                    dtype=np.int64)

"""
stat_3d.keys() =>  dict_keys(['std', 'dim_use', 'train', 'test', 'mean'])
std => (96., )
mean => (96.,)
dim_use => (48, ) ?????
train => dict{[user, action, camera_id]} ex) dict{[6, 'Walking', 'Walking 1.60457274.h5']} // data = int // len 600 = 15 actions * 8 cameras+extra_actions * 5 users
test => same as train, user = 9, 11 // len 240
(7,
 'Photo',
 'Photo 1.58860488.h5'): array([[514.54570615, -606.40670751, 5283.29114444],
                                [513.19690503, -606.27874917, 5282.94296128],
                                [511.72623278, -606.3556718, 5282.09161439],
                                ...,
                                [660.21544235, -494.87670603, 5111.48298849],
                                [654.79473179, -497.67942449, 5111.05843265],
                                [649.61962945, -498.74291164, 5111.91590807]])}

"""


# actions = ["Directions",
#            "Discussion",
#            "Eating",
#            "Greeting",
#            "Phoning",
#            "Photo",
#            "Posing",
#            "Purchases",
#            "Sitting",
#            "SittingDown",
#            "Smoking",
#            "Waiting",
#            "WalkDog",
#            "Walking",
#            "WalkTogether"]
# actions = ["Photo"]
# test

class Human36M(Dataset):
    def __init__(self, actions, data_path, set_num_samples, use_hg=True, is_train=True):
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
            self.train_3d = torch.load(os.path.join(data_path, 'train_3d.pth.tar')) ## (48,)
            self.train_2d = torch.load(os.path.join(data_path, train_2d_file)) ## (32, )

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
            self.test_3d = torch.load(os.path.join(data_path, 'test_3d.pth.tar'))
            self.test_2d = torch.load(os.path.join(data_path, test_2d_file))

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

