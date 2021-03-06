import torch
import numpy as np
from multiprocessing import Pool
from blitz.modules import TrainableRandomDistribution


def my_relu(arr):
    arr = arr * (arr > 0)
    return arr


model_path = "ERR - NO MODEL SET. Call set_model_path function."


def set_model_path(m):
    global model_path
    model_path = m
    pass


GLOBAL_samples = "lollol"


def gen_samples(iters):
    global GLOBAL_samples
    ## 导出模型的权重和偏置
    loaded_model = np.load(model_path, allow_pickle=True)
    # 权重，偏差，均值，标准差
    [fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, fc4_w_mu, fc4_b_mu, fc4_w_rho, fc4_b_rho] = \
        loaded_model['arr_0'], loaded_model['arr_1'], \
        loaded_model['arr_2'], loaded_model['arr_3'], loaded_model['arr_4'], loaded_model['arr_5'], \
        loaded_model['arr_6'], loaded_model['arr_7'], loaded_model['arr_8'], loaded_model['arr_9']
    fc4_w, fc4_b = [], []
    for i in range(iters):
        weight_sampler = TrainableRandomDistribution(torch.from_numpy(fc4_w_mu), torch.from_numpy(fc4_w_rho))
        fc4_w.append(weight_sampler.sample().detach().numpy())
        bias_sampler = TrainableRandomDistribution(torch.from_numpy(fc4_b_mu), torch.from_numpy(fc4_b_rho))
        fc4_b.append(bias_sampler.sample().detach().numpy())
    fc4_w, fc4_b = np.array(fc4_w), np.array(fc4_b)
    GLOBAL_samples = [fc4_w, fc4_b]
    pass


def propagate_interval(W, W_std, b, b_std, x_l, x_u, eps):
    """
    Interval propogation in weight and input space.
    @Variable W - the sampled weight that gave an output in the valid output region
    @Variable qW - the variational posterior weight matrix values (mean and variance)
    @Variable b - the sampled bias that gave an output in the valid output region
    @Variable qb - the variational posterior bias vector values (mean and variance)
    @Variable x_l - the lower bound of the input region
    @Variable x_u - the upper bound of the input region
    @Variable eps - the margin to propagate in the weight space (we add and subtract this value)
    """
    W_l, W_u = W - (eps * W_std), W + (eps * W_std)  # Use eps as small symetric difference about the mean
    b_l, b_u = b - (eps * b_std), b + (eps * b_std)  # Use eps as small symetric difference about the mean
    h_max = np.zeros(len(W[0]))  # Placeholder variable for return value
    h_min = np.zeros(len(W[0]))  # Placeholder variable for return value
    for i in range(len(W)):  # This is literally just a step-by-step matrix multiplication
        for j in range(len(W[0])):  # where we are taking the min and max of the possibilities
            out_arr = [W_l[i][j] * x_l[i], W_l[i][j] * x_u[i],
                       W_u[i][j] * x_l[i], W_u[i][j] * x_u[i]]
            h_min[j] += min(out_arr)
            h_max[j] += max(out_arr)
    h_min = h_min + b_l
    h_max = h_max + b_u
    return h_min, h_max  # Return the min and max of the intervals.
    # (dont forget to apply activation function after)


# Code for merging overlapping intervals. Taken from here:
# https://stackoverflow.com/questions/49071081/merging-overlapping-intervals-in-python
# This function simple takes in a list of intervals and merges them into all 
# continuous intervals and returns that list 
def merge_intervals(intervals):
    sorted_intervals = sorted(intervals)
    interval_index = 0
    intervals = np.asarray(intervals)
    for i in sorted_intervals:
        if i[0] > sorted_intervals[interval_index][1]:
            interval_index += 1
            sorted_intervals[interval_index] = i
        else:
            sorted_intervals[interval_index] = [sorted_intervals[interval_index][0], i[1]]
    return sorted_intervals[:interval_index + 1]


import math
from scipy.special import erf


def compute_erf_prob(intervals, mean, stddev):
    """
    Given a set of disjoint intervals, compute the probability of a random
    sample from a guassian falling in these intervals. (Taken from lemma)
    of the document
    """
    prob = 0.0
    for interval in intervals:
        val1 = erf((mean - interval[0]) / (math.sqrt(2) * (stddev)))
        val2 = erf((mean - interval[1]) / (math.sqrt(2) * (stddev)))
        prob += 0.5 * (val1 - val2)
    return prob


def compute_single_prob(arg):
    vector_intervals, marg, mean, std = arg
    intervals = []
    for num_found in range(len(vector_intervals)):
        # !*! Need to correct and make sure you scale margin
        interval = [vector_intervals[num_found] - (std * marg), vector_intervals[num_found] + (std * marg)]
        intervals.append(interval)
    p = compute_erf_prob(merge_intervals(intervals), mean, std)
    if (p == 0.0):
        print("error")
        return float("-inf")
    return p


def interval_bound_propagation(a):
    """
    Probabalistic Reachability for Bayesian Neural Networks - V 0.0.1 - Variable Margin
    @Variable x - the original input (not used, may delete)
    @Variable in_reg - a list [x_l, x_u] containing the region of interest in the input space
    @Variable out_reg - a list [y_l, y_u] containing the region of interest in the output space
    @Variable w_margin - a float value dictating the amount to add and subtract to create compact
                         set in the weight space given some valid sample from weight space
    @Variable search_samps - number of posterior samples to take in order to check for valid
                             samples (i.e. samples that cause output to be in valid range)

    @Return - A valid lowerbound on the probability that the input region causes BNN  to give
              ouput bounded by the output region. Converges to exact solution when margin is
              small and samples goes to infinity.
    """
    global y
    x, in_reg, out_maximal, w_margin, search_samps, id = a
    reverse = False
    x = np.asarray(x)
    x = x.astype('float64')
    relu3_l, relu3_u = my_relu(np.array(in_reg[0])), my_relu(np.array(in_reg[1]))
    out_ind = out_maximal
    loaded_model = np.load(model_path, allow_pickle=True)
    # 权重，偏差，均值，标准差
    [fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, fc4_w_mu, fc4_b_mu, fc4_w_rho, fc4_b_rho] = \
        loaded_model['arr_0'], loaded_model['arr_1'], \
        loaded_model['arr_2'], loaded_model['arr_3'], loaded_model['arr_4'], loaded_model['arr_5'], \
        loaded_model['arr_6'], loaded_model['arr_7'], loaded_model['arr_8'], loaded_model['arr_9']
    # First, sample and hope some weights satisfy the out_reg constraint
    [fc4_w, fc4_b] = GLOBAL_samples
    fc4_w = fc4_w[id * search_samps: (id + 1) * search_samps]
    fc4_b = fc4_b[id * search_samps: (id + 1) * search_samps]
    valid_weight_intervals = []
    err = 0
    x1 = my_relu(np.matmul(x, fc1_w) + fc1_b)
    x2 = my_relu(np.matmul(x1, fc2_w) + fc2_b)
    x3 = my_relu(np.matmul(x2, fc3_w) + fc3_b)
    for i in range(search_samps):
        # Full forward pass in one line :-)
        y = np.matmul(x3, fc4_w[i]) + fc4_b[i]
        # Logical check if weights sample out_reg constraint
        pre_ind = np.argmax(y)
        extra_gate = (reverse and pre_ind != out_ind)
        if (pre_ind == out_ind or extra_gate):
            # If so, do interval propagation
            # 输出y的区间
            y_pred_l, y_pred_u = propagate_interval(fc4_w[i], fc4_w_rho, fc4_b[i], fc4_b_rho, relu3_l, relu3_u, w_margin)
            assert ((y_pred_l <= y).all())
            assert ((y_pred_u >= y).all())
            # Check if interval propagation still respects out_reg constraint
            safety_check = True
            value_ind = 0
            value_l = y_pred_l[out_ind]
            for value_u in y_pred_u:
                if value_l < value_u and value_ind != out_ind:  # 如果 最终输出的下界<第i个预测的上界 且 i!=最终输出的索引: 则不安全
                    safety_check = False
                    break
                value_ind += 1
            if safety_check:
                # If it does, add the weight to the set of valid weights
                valid_weight_intervals.append([fc4_w[i], fc4_b[i]])
        else:
            err += 1
            # print(y, np.argmax(y))
            # print("Hm, incorrect prediction is worrying...")
            continue
    print("We found %s many valid intervals." % (len(valid_weight_intervals)))
    print("Pred error rate: %s/%s=%s" % (err, search_samps, (err / float(search_samps))))
    if (len(valid_weight_intervals) == 0):
        return 0.0
    # Now we need to take all of the valid weight intervals we found and merge them
    # so we seperate the valid intervals into their respective variables
    """
    vW_0, vb_0, vW_1, vb_1 = [], [], [], []
    for v in valid_weight_intervals:
        #np.asarray(v[0]) i removed this... should i not have? -MW
        vW_0.append(v[0])
        vb_0.append(v[1])
        vW_1.append(v[2])
        vb_1.append(v[3])
    """
    return valid_weight_intervals

def compute_all_intervals_proc(a):
    V, isweight, i, margin, numproc = a
    print("STARTING COMPUTE FOR %s, %s" % (i, isweight))

    loaded_model = np.load(model_path, allow_pickle=True)
    # 权重，偏差，均值，标准差
    [fc1_w, fc1_b, fc2_w, fc2_b, mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = \
        loaded_model['arr_0'], loaded_model['arr_1'], loaded_model['arr_2'], loaded_model['arr_3'], \
        loaded_model['arr_4'], loaded_model['arr_5'], loaded_model['arr_6'], loaded_model['arr_7'], \
        loaded_model['arr_8'], loaded_model['arr_9'], loaded_model['arr_10'], loaded_model['arr_11']

    vW_0 = []
    for valid_intervals in V:
        vW_0.append(valid_intervals[i])

    vW_0 = np.asarray(vW_0)

    print("交换维度前: ", vW_0.shape)
    if (isweight):
        vW_0 = np.swapaxes(vW_0, 0, 2)
        vW_0 = np.swapaxes(vW_0, 0, 1)
    else:
        vW_0 = np.swapaxes(vW_0, 0, 1)
    print("交换维度后: ", vW_0.shape)
    # After we merge them, we need to use the erf function to evaluate exactly what the
    # lower bound on the probability is!
    ind = i
    nproc = numproc
    print("Using %s processes" % (nproc))
    p = Pool(nproc)

    # Need something more general here for the multilayer case
    if (ind == 0):

        print(vW_0.shape)
        pW_0 = np.ones((vW_0.shape[0], vW_0.shape[1]))
        for i in range(len(vW_0)):
            args = []
            for j in range(len(vW_0[i])):
                args.append((vW_0[i][j], margin, mW_0[i][j], dW_0[i][j]))
            arr = p.map(compute_single_prob, args)
            pW_0[i] = arr
    elif (ind == 2):
        pW_0 = np.ones((vW_0.shape[0], vW_0.shape[1]))
        for i in range(len(vW_0)):
            args = []
            for j in range(len(vW_0[i])):
                args.append((vW_0[i][j], margin, mW_1[i][j], dW_1[i][j]))
            arr = p.map(compute_single_prob, args)
            pW_0[i] = arr
    elif (ind == 1):
        pW_0 = np.ones((vW_0.shape[0]))
        args = []
        for i in range(len(vW_0)):
            args.append((vW_0[i], margin, mb_0[i], db_0[i]))
        pW_0 = p.map(compute_single_prob, args)

    elif (ind == 3):
        pW_0 = np.ones((vW_0.shape[0]))
        args = []
        for i in range(len(vW_0)):
            args.append((vW_0[i], margin, mb_1[i], db_1[i]))
        pW_0 = p.map(compute_single_prob, args)
    pW_0 = np.asarray(pW_0)
    print("HERE IS THE SHAPE")
    print(pW_0.shape)
    p.close()
    p.join()
    p = 0.0
    for i in pW_0.flatten():
        try:
            p += math.log(i)
        except:
            return float("-inf")
    return p


