
import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

DIM_USE2 = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15, 16, 17, 24, 25, 26,
       27, 30, 31, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 54, 55],
      dtype=np.int64)

DIM_USE3 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22,
                       23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58,
                       59, 75, 76, 77, 78, 79, 80, 81, 82, 83],
                    dtype=np.int64)

# torch/serialization.py
#
# python2에서 생성된 pth.tar를 불러오는 경우
#
# # _sys_info = pickle_module.load(f, **pickle_load_args)
# # unpickler = pickle_module.Unpickler(f, **pickle_load_args)
# # unpickler.persistent_load = persistent_load
# #
# _sys_info = pickle_module.load(f, encoding='iso-8859-1')
# unpickler = pickle_module.Unpickler(f, encoding='iso-8859-1')
# unpickler.persistent_load = persistent_load


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


# data_dir = 'D:\\GK4D\\HumanPose\\3d_pose_baseline_pytorch\\z_score_data_python3\\'
data_dir = 'E:\\Human36m\\z_score_data_python2\\'
data_dir_new = 'D:\\GK4D\\HumanPose\\3d_pose_baseline_pytorch\\data\\'

stat_2d = torch.load(data_dir+'stat_2d.pth.tar')
# torch.save(stat_2d, data_dir_new+'stat_2d.pth.tar')
stat_3d = torch.load(data_dir+'stat_3d.pth.tar')
# torch.save(stat_3d, data_dir_new+'stat_3d.pth.tar')

test_2d = torch.load(data_dir+'test_2d.pth.tar')
for k2d in test_2d.keys():
    d_temp = test_2d[k2d]
    d_temp = unNormalizeData(d_temp, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])  # (64, 96)
    test_2d[k2d] = d_temp[:, DIM_USE2]
    # dd = d_temp[:, DIM_USE2]
    # for i in range(dd.shape[0]):
    #     sample_joint = np.reshape(np.asarray(dd[i,:]), (16, 2))
    #     plt.figure()
    #
    #     for i in range(len(sample_joint)):
    #         x, y = sample_joint[i]
    #         plt.scatter(x, y)
    #
    #     plt.gca().invert_yaxis()

# torch.save(test_2d, data_dir_new+'test_2d.pth.tar')

train_2d = torch.load(data_dir+'train_2d.pth.tar')
for k2d in train_2d.keys():
    d_temp = train_2d[k2d]
    d_temp = unNormalizeData(d_temp, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])  # (64, 96)
    train_2d[k2d] = d_temp[:, DIM_USE2]

    dd = d_temp[:, DIM_USE2]
    # for i in range(dd.shape[0]):
    #     sample_joint = np.reshape(np.asarray(dd[i,:]), (16, 2))
    #     plt.figure()
    #
    #     for i in range(len(sample_joint)):
    #         x, y = sample_joint[i]
    #         plt.scatter(x, y)
    #
    #     plt.gca().invert_yaxis()
    # print(1)
torch.save(train_2d, data_dir_new+'train_2d.pth.tar')


test_3d = torch.load(data_dir+'test_3d.pth.tar')
for k3d in test_3d.keys():
    d_temp = test_3d[k3d]
    d_temp = unNormalizeData(d_temp, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])  # (64, 96)
    test_3d[k3d] = d_temp[:, DIM_USE3]
torch.save(test_3d, data_dir_new+'test_3d.pth.tar')

train_3d = torch.load(data_dir+'train_3d.pth.tar')
for k3d in train_3d.keys():
    d_temp = train_3d[k3d]
    d_temp = unNormalizeData(d_temp, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])  # (64, 96)
    train_3d[k3d] = d_temp[:, DIM_USE3]

    # dd = d_temp[:, DIM_USE3]
    # for i in range(dd.shape[0]):
    #     sample_joint = np.reshape(np.asarray(dd[i,:]), (16, 3))
    #     plt.figure()
    #
    #     for i in range(len(sample_joint)):
    #         x, y, z = sample_joint[i]
    #         plt.scatter(x, y)
    #
    #     plt.gca().invert_yaxis()
    # print(1)
torch.save(train_3d, data_dir_new+'train_3d.pth.tar')
