import logging
import time
import numpy as np
from d3b1_deeppoly.verify.deepPoly import network


def deepPoly_interval(rlv, mnist, epsilon=0.025):
    """
    输入:
        rlv: 预训练的模型
        property: 测试点，MNIST中的一张图片
        epsilon: 噪声强度
    输出:
        x_l, x_u: 输入点x的区间
        relu2_l, relu2_u: 用deepPoly求出的第二个relu层区间
    """
    net = network()
    net.load_rlv(rlv)
    net.clear()
    # 测试点的区间
    net.load_robustness(mnist, epsilon, TRIM=True)
    # 用deepPoly求出的区间
    y_l, y_u = net.deeppoly()
    return y_l, y_u

def main(name):
    epsilon = 0.025

    for image in range(2, 3):
        print("image: ", image)
        rlv = './deepPoly/rlv/%s.rlv' % name  # deepPoly所需的权重文件
        mnist = '../../mnist/mnist_%s_local_property.in' % image
        y_l, y_u = deepPoly_interval(rlv, mnist, epsilon)
        print(y_l, y_u)
        print(np.argmax(y_l))

if __name__ == "__main__":
    logging.info(time.time())
    main('HybridNN_d3b1_256_128_64_epochs3')