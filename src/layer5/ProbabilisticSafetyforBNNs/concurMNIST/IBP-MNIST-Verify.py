# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--imnum")
# parser.add_argument("--eps")
# parser.add_argument("--samples")
# parser.add_argument("--width")
# parser.add_argument("--margin")
#
# args = parser.parse_args()
# image = int(args.imnum)
# epsilon = float(args.eps)
# iters = int(args.samples)
# width = int(args.width)
# margin = float(args.margin)

# python IBP-MNIST-Verify-Test.py --imnum $IMNUM --eps 0.025 --samples 1250 --width 64 --margin 2.0
image = 10  # 第i张图
epsilon = 0.025  # 噪声
iters = 500
# iters = 5
width = 64
margin = 2.0

nproc = 25
# nproc = 5

# pickle提供了一个简单的持久化功能, 可以将对象以文件的形式存放在磁盘上。
import pickle
import math
import numpy as np
import tensorflow as tf
import ProbablisticReachability
from ProbablisticReachability import compute_all_intervals_proc
from ProbablisticReachability import interval_bound_propagation_VCAS


def my_relu(arr):
    arr = arr * (arr > 0)
    return arr


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_test = X_test / 255.
X_test = X_test.astype("float32").reshape(-1, 28 * 28)

y_test = tf.one_hot(y_test, 10)

x = X_test[image]
x1 = x
inp = np.asarray(X_test[image])
x_u = np.clip(inp + epsilon, 0, 1)  # 限制在（0,1）
x_l = np.clip(inp - epsilon, 0, 1)

# 模拟一层BNN的输出
## 导出一层BNN模型的权重和偏置
model_path = "MNIST_Networks/VIMODEL_MNIST_1_%s_relu.net.npz" % width
try:
    loaded_model = np.load(model_path, allow_pickle=True)
    [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
except:
    with open(model_path, 'rb') as pickle_file:
        [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = pickle.load(pickle_file)  # 均值、标准差

## 对权重和偏置进行采样，生成search_samps个一层BNN的权重和偏置
# First, sample and hope some weights satisfy the out_reg constraint
search_samps = 150
sW_0 = np.random.normal(mW_0, dW_0, (search_samps, mW_0.shape[0], mW_0.shape[1]))
sb_0 = np.random.normal(mb_0, db_0, (search_samps, mb_0.shape[0]))
sW_1 = np.random.normal(mW_1, dW_1, (search_samps, mW_1.shape[0], mW_1.shape[1]))
sb_1 = np.random.normal(mb_1, db_1, (search_samps, mb_1.shape[0]))

## 前向传播
y = np.zeros(10)
for i in range(search_samps):
    y += (np.matmul(my_relu(np.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i])

print("Mean prediction")
print(y / float(search_samps))

x_reg_1 = [x_l, x_u]
out_cls = np.argmax(y)

#
ProbablisticReachability.set_model_path(model_path)
## 对权重和偏置进行采样，生成iters个一层BNN的权重和偏置
ProbablisticReachability.gen_samples(iters)
import time

start = time.time()
from multiprocessing import Pool
## 得到最大安全权重集(权重的区间)
p = Pool(nproc)
args = []
for i in range(nproc):
    args.append((x1, x_reg_1, out_cls, margin, int(iters / nproc), i))
valid_intervals = p.map(interval_bound_propagation_VCAS, args)
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
vW_0 = []
vb_0 = []
vW_1 = []
vb_1 = []
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
print("IN TOTAL THERE ARE THIS MANY INTERVALS: ")
print(len(valid_intervals))
if len(valid_intervals) == 0:
    print("最大安全权重集为空")
    assert len(valid_intervals) != 0

"""
pW_0 = ProbablisticReachability.compute_interval_probs_weight(np.asarray(vW_0), marg=margin, mean=mW_0, std=dW_0)
pb_0 = ProbablisticReachability.compute_interval_probs_bias(np.asarray(vb_0), marg=margin, mean=mb_0, std=db_0)
pW_1 = ProbablisticReachability.compute_interval_probs_weight(np.asarray(vW_1), marg=margin, mean=mW_1, std=dW_1)
pb_1 = ProbablisticReachability.compute_interval_probs_bias(np.asarray(vb_1), marg=margin, mean=mb_1, std=db_1)


p = 0.0
for i in pW_0.flatten():
    p+=math.log(i)
for i in pb_0.flatten():
    p+=math.log(i)
for i in pW_1.flatten():
    p+=math.log(i)
for i in pb_1.flatten():
    p+=math.log(i)
p = math.exp(p)
ph1 = p
"""
# valid_intervals =  interval_bound_propagation_VCAS(x1, x_reg_1, out_cls, w_margin=margin, search_samps=iters)
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
