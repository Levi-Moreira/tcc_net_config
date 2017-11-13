import numpy as np
import cv2

def maxpool_forward(X, size=2, stride=2):
    def maxpool(X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    return _pool_forward(X, maxpool, size, stride)


def conv_forward(X, W, b, stride=1, padding=1):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = np.matmul(W_col,X_col) + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out, cache

def dropout_forward(X, p_dropout):
    u = np.random.binomial(1, p_dropout, size=X.shape) / p_dropout
    out = X * u
    cache = u
    return out, cache

def relu_forward(X):
    out = np.maximum(X, 0)
    cache = X
    return out, caches

def fc_forward(X, W, b):
    out = np.matmul(X,W) + b
    cache = (W, X)
    return out, cache

def bn_forward(X, gamma, beta, cache, momentum=.9, train=True):
    running_mean, running_var = cache

    if train:
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_norm = (X - mu) / np.sqrt(var + c.eps)
        out = gamma * X_norm + beta

        cache = (X, X_norm, mu, var, gamma, beta)

        running_mean = util.exp_running_avg(running_mean, mu, momentum)
        running_var = util.exp_running_avg(running_var, var, momentum)
    else:
        X_norm = (X - running_mean) / np.sqrt(running_var + c.eps)
        out = gamma * X_norm + beta
        cache = None

    return out, cache, running_mean, running_var

def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new

def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T




def main():

    X = cv2.imread("/media/levi/E/dataset/data/pos/image_movie01_015_0.png")
    W1 = np.fromfile('/media/levi/E/dataset/ResultModel/net_data/w_conv2.npy')
    b1 = np.fromfile('/media/levi/E/dataset/ResultModel/net_data/b_conv1.npy')
    print np.shape(W1)
    print len(b1)
    # out_conv_1, chache_conv_1 = conv_forward(X, W1, b1, 4, 0)
    # out_conv_1, chache_conv_1 = relu_forward(out_conv_1)
    #
    # hpool, hpool_cache = maxpool_forward(out_conv_1)
    # out_max_pool1 = hpool.ravel().reshape(X.shape[0], -1)


main()
