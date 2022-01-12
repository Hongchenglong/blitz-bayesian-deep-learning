import torch
import numpy as np
from src.layer3.train.HybridNN import HybridNN

net = HybridNN()
name = 'HybridNN_layer3_epochs1'
PATH = '../train/pth/%s.pth' % name
pre_weights = torch.load(PATH)

# print("rlv start")
# f = open("./deepPoly/rlv/%s.pth.rlv" % name, "a")
#
# # 输入层
# f.write("# Layer 0 784 Input data\n")
# for i in range(784):
#     f.write("Input inX%s\n" % str(i))
#
# # 前2层隐藏层
# start = 1; end = 3
# for i in range(start, end):
#     bias = pre_weights['fc%d.bias' % i]
#     weight = pre_weights['fc%d.weight' % i]
#     width = bias.shape[0]
#     f.write("# Layer %d %d ReLU relu%d\n" % (i, width, i))
#     for wid in range(width):
#         f.write("ReLU relu%dX%d %f" % (i, wid, bias[wid]))  # 第i层，宽度，偏置b
#         for wei in range(weight.shape[1]):
#             if i == 1:
#                 f.write(" %f inX%d" % (weight[wid][wei], wei))
#             else:
#                 f.write(" %f relu%dX%d" % (weight[wid][wei], i-1, wei))
#         f.write("\n")
#
# # 输出层
# f.write("# Layer %d 10 Linear res\n" % end)
# bias = pre_weights['fc%d.bias_mu' % end]
# weight = pre_weights['fc%d.weight_mu' % end]
# width = bias.shape[0]
# for wid in range(width):
#     f.write("Linear resX%d %f" % (wid, bias[wid]))  # 宽度，偏置b
#     for wei in range(weight.shape[1]):
#         f.write(" %f relu%dX%d" % (weight[wid][wei], end-1, wei))
#     f.write("\n")
#
# # 预测
# f.write("# Layer %d 10 Linear Accuracy\n" % (end+1))
# for i in range(10):
#     f.write("Linear outX%d 0.0 1.0 resX%d\n" % (i, i))
# f.close()
# print("rlv end")

print("npz start")
np.savez("./IBP/npz/%s.npz" % name,
         np.asarray(pre_weights['fc1.weight'].T, dtype='float32'), np.asarray(pre_weights['fc1.bias']),
         np.asarray(pre_weights['fc2.weight'].T, dtype='float32'), np.asarray(pre_weights['fc2.bias'], dtype='float32'),
         np.asarray(pre_weights['fc3.weight_mu'].T, dtype='float32'), np.asarray(pre_weights['fc3.weight_rho'].T, dtype='float32'),
         np.asarray(pre_weights['fc3.bias_mu'].T, dtype='float32'), np.asarray(pre_weights['fc3.bias_rho'], dtype='float32'))
print("npz end")


