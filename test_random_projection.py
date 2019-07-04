
import torch
import numpy as np
import math


num_data = 3
# random projection
az_angle = np.random.uniform(-math.pi, math.pi, num_data)
el_angle = np.random.uniform(-math.pi/9, math.pi/9, num_data)
random_rot = torch.zeros(num_data, 3, 3)
random_rot_inv = torch.zeros(num_data, 3, 3)
for k in range(num_data):
    Theta = math.acos(math.cos(el_angle[k]) * math.cos(az_angle[k]))
    Phi = math.atan(math.tan(el_angle[k]) / math.sin(az_angle[k]))
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

    random_rot[k, :, :] = torch.mm(Ry, Rx)
    random_rot_inv[k, :, :] = torch.inverse(random_rot[k, :, :])


input = torch.randn(3, 16 * 3)



pred3d_reshape = input.reshape((-1, 16, 3)).permute((0, 2, 1))
pred3d_origin = pred3d_reshape - pred3d_reshape[:, :, 0].reshape((-1, 3, 1))

pred3d_transform = torch.matmul(random_rot, pred3d_origin)
pred3d_transform_inv = torch.matmul(random_rot_inv, pred3d_transform)

# pred3d_transform_reshape = pred3d_transform.reshape(-1, 16 * 3)
# pred3d_transform_reshape = pred3d_transform_reshape.reshape((-1, 16, 3))
#
# pred3d_transform = torch.matmul(random_rot_inv, pred3d_transform_reshape.permute((0, 2, 1))).permute((0, 2, 1))


# print(pred3d_transform_inv - pred3d_transform)
print(pred3d_origin - pred3d_transform_inv)
# print(torch.matmul(random_rot, pred3d_origin.permute((0, 2, 1))).permute((0, 2, 1)))
# print(pred3d_transform_inv)
# print(torch.matmul(random_rot[k, :, :], random_rot_inv[k, :, :]))
