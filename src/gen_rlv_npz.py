from src.HybridNN import HybridNN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms


net = HybridNN()
PATH = '../NNs/HybridNN_layer3_epochs1.pth'
pre_weights = torch.load(PATH)

f = open("DeepPoly/rlv/HybridNN_layer3_epochs1.pth.rlv", "a")

# 输入层
f.write("# Layer 0 784 Input data\n")
for i in range(784):
    f.write("Input inX%s\n" % str(i))

# 前2层隐藏层
start = 1; end = 3
for i in range(start, end):
    bias = pre_weights['fc%d.bias' % i]
    weight = pre_weights['fc%d.weight' % i]
    width = bias.shape[0]
    f.write("# Layer %d %d ReLU relu%d\n" % (i, width, i))
    for wid in range(width):
        f.write("ReLU relu%dX%d %f" % (i, wid, bias[wid]))  # 第i层，宽度，偏置b
        for wei in range(weight.shape[1]):
            if i == 1:
                f.write(" %f inX%d" % (weight[wid][wei], wei))
            else:
                f.write(" %f relu%dX%d" % (weight[wid][wei], i-1, wei))
        f.write("\n")

# 输出层
f.write("# Layer %d 10 Linear res\n" % end)
bias = pre_weights['fc%d.bias_mu' % end]
weight = pre_weights['fc%d.weight_mu' % end]
width = bias.shape[0]
for wid in range(width):
    f.write("Linear resX%d %f" % (wid, bias[wid]))  # 宽度，偏置b
    for wei in range(weight.shape[1]):
        f.write(" %f relu%dX%d" % (weight[wid][wei], end-1, wei))
    f.write("\n")

# 预测
f.write("# Layer %d 10 Linear Accuracy\n" % (end+1))
for i in range(10):
    f.write("Linear outX%d 0.0 1.0 resX%d\n" % (i, i))

f.close()

# print("npz")
# np.savez("HybridNN_epoch1.npz", pre_weights['fc1.weight'].T, pre_weights['fc1.bias'],
#                                 pre_weights['fc2.weight'].T, pre_weights['fc2.bias'],
#                                 # pre_weights['fc3.weight'].T, pre_weights['fc3.bias'],
#                                 # pre_weights['fc4.weight'].T, pre_weights['fc4.bias'],
#                                 pre_weights['fc3.weight_mu'].T, pre_weights['fc3.weight_rho'],
#                                 pre_weights['fc3.bias_mu'].T, pre_weights['fc3.bias_rho'])
print("over")


