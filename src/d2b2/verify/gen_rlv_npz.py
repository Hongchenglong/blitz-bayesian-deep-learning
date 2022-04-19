import os
import torch
import numpy as np
from src.d2b2.train.HybridNN import HybridNN

net = HybridNN()
name = 'width64_epochs1'
PATH = '../train/pth/%s.pth' % name
pre_weights = torch.load(PATH)

# rlv保存deepPoly方法需要的权重
print("rlv start")
rlv = "./deepPoly/rlv/%s.rlv" % name
if not os.path.exists(rlv):
    os.mknod(rlv)
f = open(rlv, 'w')

# 输入层
f.write("# Layer 0 784 Input data\n")
for i in range(784):
    f.write("Input inX%s\n" % str(i))

# 前end-1层隐藏层
start = 1; end = 4
for i in range(start, end):
    try:
        bias = pre_weights['fc%d.bias' % i]
        weight = pre_weights['fc%d.weight' % i]
    except:
        bias = pre_weights['fc%d.bias_mu' % i]
        weight = pre_weights['fc%d.weight_mu' % i]
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
print("rlv end")

# npz保存IBP方法需要的权重
print("npz start")
npz = "./IBP/npz/%s.npz" % name
if not os.path.exists(npz):
    os.mknod(npz)
np.savez(npz,
         (pre_weights['fc1.weight'].T.detach().cpu().numpy()), (pre_weights['fc1.bias'].detach().cpu().numpy()),
         (pre_weights['fc2.weight'].T.detach().cpu().numpy()), (pre_weights['fc2.bias'].detach().cpu().numpy()),
         (pre_weights['fc3.weight_mu'].T.detach().cpu().numpy()), (pre_weights['fc3.bias_mu'].detach().cpu().numpy()),
         (pre_weights['fc4.weight_mu'].T.detach().cpu().numpy()), (pre_weights['fc4.bias_mu'].detach().cpu().numpy()),
         (pre_weights['fc3.weight_rho'].T.detach().cpu().numpy()), (pre_weights['fc3.bias_rho'].detach().cpu().numpy()),
         (pre_weights['fc4.weight_rho'].T.detach().cpu().numpy()), (np.exp(pre_weights['fc4.bias_rho'].detach().cpu().numpy())))
         # np.sqrt(np.log1p(np.exp(pre_weights['fc3.weight_rho'].T.detach().cpu().numpy()))), np.sqrt(np.log1p(np.exp(pre_weights['fc3.bias_rho'].detach().cpu().numpy()))),
         # np.sqrt(np.log1p(np.exp(pre_weights['fc4.weight_rho'].T.detach().cpu().numpy()))), np.sqrt(np.log1p(np.exp(pre_weights['fc4.bias_rho'].detach().cpu().numpy()))))
         # np.zeros_like(pre_weights['fc3.weight_rho'].T.detach().cpu().numpy()), np.zeros_like(pre_weights['fc3.bias_rho'].detach().cpu().numpy()),
         # np.zeros_like(pre_weights['fc4.weight_rho'].T.detach().cpu().numpy()), np.zeros_like(pre_weights['fc4.bias_rho'].detach().cpu().numpy()))
print("npz end")


