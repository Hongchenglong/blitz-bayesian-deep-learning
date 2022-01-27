import torch
from src.layer3.verify.deepPoly.DeepPoly import network
from blitz.modules import TrainableRandomDistribution


def my_relu(arr):
    arr = arr * (arr > 0)
    return arr


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
    x_l, x_u = net.load_robustness(mnist, epsilon, TRIM=True)
    # 用deepPoly求出的区间
    relu2_l, relu2_u = net.deeppoly()
    return x_l, x_u, relu2_l, relu2_u


def IBP_p(image, x, x_l, x_u, relu2_l, relu2_u, width, npz, model_path):
    """
    image: 第i张mnist图片
    width: 倒数第二层贝叶斯层的宽度
    npz: 预训练保存的weight和bias
    iters: 迭代次数
    margin: weight margin
    nproc: 进程数量
    """

    import math
    import numpy as np
    import IBP.ProbablisticReachability
    from IBP.ProbablisticReachability import compute_all_intervals_proc
    from IBP.ProbablisticReachability import interval_bound_propagation

    iters = 500
    nproc = 25
    # margin = 2.0
    margin = 2.0

    # 模拟神经网络的输出
    ## 导出模型的权重和偏置
    loaded_model = np.load(model_path, allow_pickle=True)
    # 权重，偏差，均值，标准差
    [fc1_w, fc1_b, fc2_w, fc2_b, mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = \
        loaded_model['arr_0'], loaded_model['arr_1'], loaded_model['arr_2'], loaded_model['arr_3'], \
        loaded_model['arr_4'], loaded_model['arr_5'], loaded_model['arr_6'], loaded_model['arr_7'], \
        loaded_model['arr_8'], loaded_model['arr_9'], loaded_model['arr_10'], loaded_model['arr_11']

    # 对权重和偏置进行采样，生成search_samps个一层BNN的权重和偏置
    # First, sample and hope some weights satisfy the out_reg constraint
    search_samps = 150
    sW_0, sb_0, sW_1, sb_1 = [], [], [], []
    for i in range(search_samps):
        weight_sampler = TrainableRandomDistribution(torch.from_numpy(mW_0), torch.from_numpy(dW_0))
        sW_0.append(weight_sampler.sample().detach().numpy())
        bias_sampler = TrainableRandomDistribution(torch.from_numpy(mb_0), torch.from_numpy(db_0))
        sb_0.append(bias_sampler.sample().detach().numpy())
        weight_sampler = TrainableRandomDistribution(torch.from_numpy(mW_1), torch.from_numpy(dW_1))
        sW_1.append(weight_sampler.sample().detach().numpy())
        bias_sampler = TrainableRandomDistribution(torch.from_numpy(mb_1), torch.from_numpy(db_1))
        sb_1.append(bias_sampler.sample().detach().numpy())
    sW_0 = np.array(sW_0); sb_0 = np.array(sb_0); sW_1 = np.array(sW_1); sb_1 = np.array(sb_1);

    ## 前向传播
    y = np.zeros(10)
    x1 = my_relu(np.matmul(x, fc1_w) + fc1_b)
    x2 = my_relu(np.matmul(x1, fc2_w) + fc2_b)
    for i in range(search_samps):
        y += (np.matmul(my_relu(np.matmul(x2, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i])

    out_cls = np.argmax(y)
    x_reg_1 = [relu2_l, relu2_u]

    print("Mean prediction")
    print(y / float(search_samps), out_cls)

    IBP.ProbablisticReachability.set_model_path(model_path)
    IBP.ProbablisticReachability.gen_samples(iters)

    import time

    start = time.time()
    from multiprocessing import Pool
    ## 得到最大安全权重集(权重的区间)
    p = Pool(nproc)
    args = []
    for i in range(nproc):
        args.append((x, x_reg_1, out_cls, margin, int(iters / nproc), i))
    valid_intervals = p.map(interval_bound_propagation, args)
    p.close()
    p.join()

    stop = time.time()

    elapsed = stop - start
    print("len(valid_intervals): ", len(valid_intervals))

    if len(valid_intervals) == 0:
        import logging

        ph1 = 0.0
        logging.basicConfig(filename="Runs%s.log" % width, level=logging.DEBUG)
        logging.info("image=%s,width=%s,epsilon=%s,margin=%s,iters=%s,elapsed=%s,ph1=%s"
                     % (image, width, epsilon, margin, iters, elapsed, ph1))

    vad_int = []
    logged_flag = False
    try:
        for i in range(len(valid_intervals)):
            for j in range(len(valid_intervals[i])):
                vad_int.append([valid_intervals[i][j][0],
                                valid_intervals[i][j][1],
                                valid_intervals[i][j][2],
                                valid_intervals[i][j][3]])
    except:
        import logging

        ph1 = 0.0
        logging.basicConfig(filename="Runs%s.log" % width, level=logging.DEBUG)
        logging.info("image=%s,width=%s,epsilon=%s,margin=%s,iters=%s,elapsed=%s,ph1=%s"
                     % (image, width, epsilon, margin, iters, elapsed, ph1))
        logged_flag = True
    valid_intervals = vad_int
    valid_intervals_len = len(valid_intervals)
    print("IN TOTAL THERE ARE THIS MANY INTERVALS: ", valid_intervals_len)
    if valid_intervals_len == 0:
        print("最大安全权重集为空")
        return 0
        # assert valid_intervals_len != 0

    if margin != 0:
        p1 = compute_all_intervals_proc((valid_intervals, True, 0, margin, nproc))
        p2 = compute_all_intervals_proc((valid_intervals, False, 1, margin, nproc))
        p3 = compute_all_intervals_proc((valid_intervals, True, 2, margin, nproc))
        p4 = compute_all_intervals_proc((valid_intervals, False, 3, margin, nproc))

        ph1 = p1 + p2 + p3 + p4
        ph1 = math.exp(ph1)

    else:
        ph1 = 0.0
    print("ph1: ", ph1)
    stop = time.time()

    elapsed = stop - start
    if (not logged_flag):
        import logging

        logging.basicConfig(filename="F_Runs%s.log" % (width), level=logging.DEBUG)
        logging.info("image=%s,width=%s,epsilon=%s,margin=%s,iters=%s,elapsed=%s,len(valid_intervals)=%s,ph1=%s"
                     % (image, width, epsilon, margin, iters, elapsed, len(valid_intervals), ph1))
    return ph1


def mnist_test_point(filename):
    """
    获取mnist测试图片的784个像素点
    """
    x = []
    with open(filename) as f:
        for i in range(784):
            line = f.readline()
            x.append(float(line.strip()))
    return x


if __name__ == "__main__":
    # epsilon = 0.025
    epsilon = 0.001
    width = 64

    # image = 5
    for image in range(2, 3):
        print("image: ", image)
        # name = 'HybridNN_d2b2_width%s_epochs3' % width
        name = 'HybridNN_d2b2_64_784_width64_epochs3'
        rlv = './deepPoly/rlv/%s.pth.rlv' % name  # deepPoly所需的权重文件
        mnist = '../../mnist/mnist_%s_local_property.in' % image
        x_l, x_u, relu2_l, relu2_u = deepPoly_interval(rlv, mnist, epsilon)

        x = mnist_test_point(mnist)  # 测试点
        npz = '%s.npz' % name
        model_path = "./IBP/npz/%s" % npz  # IBP所需的权重文件
        p = IBP_p(image, x, x_l, x_u, relu2_l, relu2_u, width, npz, model_path)
