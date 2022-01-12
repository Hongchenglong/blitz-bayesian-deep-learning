from src.layer3.verify.deepPoly.DeepPoly import network


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


def IBP_p(x, x_l, x_u, relu2_l, relu2_u, npz):
    """

    npz: 预训练保存的weight和bias
    iters: 迭代次数
    margin: weight margin
    nproc: 进程数量
    """

    # pickle提供了一个简单的持久化功能, 可以将对象以文件的形式存放在磁盘上。
    import pickle
    import math
    import numpy as np
    import IBP.ProbablisticReachability
    from IBP.ProbablisticReachability import compute_all_intervals_proc
    from IBP.ProbablisticReachability import interval_bound_propagation

    iters = 10
    nproc = 5
    margin = 2.0

    # 模拟一层BNN的输出
    ## 导出一层BNN模型的权重和偏置
    model_path = "./IBP/npz/%s" % npz
    loaded_model = np.load(model_path, allow_pickle=True)
    [fc1_w, fc1_b, fc2_w, fc2_b, fc3_w_mu, fc3_w_rho, fc3_b_mu, fc3_b_rho] = \
        loaded_model['arr_0'], loaded_model['arr_1'], loaded_model['arr_2'], loaded_model['arr_3'], \
        loaded_model['arr_4'], loaded_model['arr_5'], loaded_model['arr_6'], loaded_model['arr_7']

    # 对权重和偏置进行采样，生成search_samps个一层BNN的权重和偏置
    # First, sample and hope some weights satisfy the out_reg constraint
    search_samps = 500
    fc3_w = np.random.normal(fc3_w_mu, fc3_w_rho ** 2, (search_samps, fc3_w_mu.shape[0], fc3_w_mu.shape[1]))
    fc3_b = np.random.normal(fc3_b_mu, fc3_b_rho ** 2, (search_samps, fc3_b_mu.shape[0]))

    ## 前向传播
    y = np.zeros(10)
    x1 = my_relu(np.matmul(x, fc1_w) + fc1_b)
    x2 = my_relu(np.matmul(x1, fc2_w) + fc2_b)
    for i in range(search_samps):
        y += np.matmul(x2, fc3_w[i]) + fc3_b[i]

    out_cls = np.argmax(y)
    x_reg_1 = [x_l, x_u]

    print("Mean prediction")
    print(y / float(search_samps), out_cls)


    #
    IBP.ProbablisticReachability.set_model_path(model_path)
    ## 对权重和偏置进行采样，生成iters个一层BNN的权重和偏置
    IBP.ProbablisticReachability.gen_samples(iters)

    import time

    start = time.time()
    from multiprocessing import Pool
    ## 得到最大安全权重集(权重的区间)
    p = Pool(nproc)
    args = []
    for i in range(nproc):
        args.append((x1, x_reg_1, relu2_l, relu2_u, out_cls, margin, int(iters / nproc), i))
    valid_intervals = p.map(interval_bound_propagation, args)
    p.close()
    p.join()

    stop = time.time()
    #
    # elapsed = stop - start
    # print("len(valid_intervals): ", len(valid_intervals))
    #
    # if len(valid_intervals) == 0:
    #     import logging
    #
    #     ph1 = 0.0
    #     logging.basicConfig(filename="Runs%s.log" % width, level=logging.DEBUG)
    #     logging.info("image=%s,width=%s,epsilon=%s,margin=%s,iters=%s,elapsed=%s,ph1=%s"
    #                  % (image, width, epsilon, margin, iters, elapsed, ph1))
    #
    # vad_int = []
    # logged_flag = False
    # try:
    #     for i in range(len(valid_intervals)):
    #         for j in range(len(valid_intervals[i])):
    #             vad_int.append([valid_intervals[i][j][0],
    #                             valid_intervals[i][j][1],
    #                             valid_intervals[i][j][2],
    #                             valid_intervals[i][j][3]])
    # except:
    #     import logging
    #
    #     ph1 = 0.0
    #     logging.basicConfig(filename="Runs%s.log" % width, level=logging.DEBUG)
    #     logging.info("image=%s,width=%s,epsilon=%s,margin=%s,iters=%s,elapsed=%s,ph1=%s"
    #                  % (image, width, epsilon, margin, iters, elapsed, ph1))
    #     logged_flag = True
    # valid_intervals = vad_int
    # print("IN TOTAL THERE ARE THIS MANY INTERVALS: ")
    # print(len(valid_intervals))
    # if len(valid_intervals) == 0:
    #     print("最大安全权重集为空")
    #     assert len(valid_intervals) != 0
    #
    # if margin != 0:
    #     p1 = compute_all_intervals_proc((valid_intervals, True, 0, margin, nproc))
    #     p2 = compute_all_intervals_proc((valid_intervals, False, 1, margin, nproc))
    #     p3 = compute_all_intervals_proc((valid_intervals, True, 2, margin, nproc))
    #     p4 = compute_all_intervals_proc((valid_intervals, False, 3, margin, nproc))
    #
    #     ph1 = p1 + p2 + p3 + p4
    #     ph1 = math.exp(ph1)
    #
    # else:
    #     ph1 = 0.0
    # print("ph1: ", ph1)
    # stop = time.time()
    #
    # elapsed = stop - start
    # if (not logged_flag):
    #     import logging
    #
    #     logging.basicConfig(filename="F_Runs%s.log" % (width), level=logging.DEBUG)
    #     logging.info("image=%s,width=%s,epsilon=%s,margin=%s,iters=%s,elapsed=%s,len(valid_intervals)=%s,ph1=%s"
    #                  % (image, width, epsilon, margin, iters, elapsed, len(valid_intervals), ph1))


def mnist_test_point(filename):
    x = []
    with open(filename) as f:
        for i in range(784):
            line = f.readline()
            x.append(float(line.strip()))
    return x


if __name__ == "__main__":
    epsilon = 0.025
    rlv = './deepPoly/rlv/HybridNN_layer3_epochs1.pth.rlv'
    mnist = './deepPoly/mnist/mnist_0_local_property.in'
    x = mnist_test_point(mnist)
    x_l, x_u, relu2_l, relu2_u = deepPoly_interval(rlv, mnist, epsilon)

    npz = "HybridNN_layer3_epochs1.npz"
    IBP_p(x, x_l, x_u, relu2_l, relu2_u, npz)
