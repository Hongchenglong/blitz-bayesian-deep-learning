import os
import torch

from blitz.modules import TrainableRandomDistribution


def gen_rlv(name):
    PATH = '../train/pth/%s.pth' % name
    # pre_weights = torch.load(PATH)
    pre_weights = torch.load(PATH, map_location=torch.device('cpu'))

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

    start = 1; end = 5
    for i in range(start, end):
        try:
            bias = pre_weights['fc%d.bias' % i]
            weight = pre_weights['fc%d.weight' % i]
        except:
            # bias = pre_weights['fc%d.bias_mu' % i]
            # weight = pre_weights['fc%d.weight_mu' % i]
            weight_sampler = TrainableRandomDistribution(pre_weights['fc%d.weight_sampler.mu' % i],
                                                         pre_weights['fc%d.weight_sampler.rho' % i])
            weight = weight_sampler.sample()

            bias_sampler = TrainableRandomDistribution(pre_weights['fc%d.bias_sampler.mu' % i],
                                                       pre_weights['fc%d.bias_sampler.rho' % i])
            bias = bias_sampler.sample()

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

    # 预测
    f.write("# Layer %d 10 Linear Accuracy\n" % (end+1))
    for i in range(10):
        f.write("Linear outX%d 0.0 1.0 resX%d\n" % (i, i))
    f.close()
    print(rlv, "\nrlv end")

if __name__=="__main__":
    gen_rlv("HybridNN_d3b1_256_128_64_epochs3")
