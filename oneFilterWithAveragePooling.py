import numpy as np
from scipy.optimize import minimize


np.random.seed(0)  # for reproducability

D = 64  # ambient dimension
FILTER_SIZE = 8  # size of the convolution filter
STRIDE = 1  # stride of the convolution filter
W0 = np.random.randn(FILTER_SIZE)  # true solution
N_TEST = 10000  # number of test elements


def datagen(n_samples):  # random data generation
    x = np.random.randn(n_samples, D)
    return x, F(x)


def stack(mat):
    '''
    Here we will stack an input vector so that we can perform convolution.
    '''
    # output shape: (numel,FILTER_SIZE)
    numel = (mat.shape[-1]-FILTER_SIZE+1) // STRIDE
    return np.stack([mat[..., i*STRIDE: i*STRIDE + FILTER_SIZE]
                     for i in range(numel)], axis=1)


def F(x):  # true relation between input and output
    return np.average(np.dot(stack(x), W0), axis=-1) + 1e-2*np.random.randn(x.shape[0])


def F_FNN(x, w):  # FNN function
    return np.matmul(x, w)


def F_CNN(x, w):  # CNN function
    return np.average(np.dot(stack(x), w), axis=-1)


def loss(y_true, y_pred):
    return np.sqrt(np.average((y_true-y_pred)**2))


# to store the error values
plot_range = range(500, 2001, 500)  # for number of samples
cnn_err = []
fnn_err = []

for n in plot_range:
    # generating train data
    x_train, y_train = datagen(n)

    # To obtain empirical estimate of solution,
    # we define the optimizing loss functions (which varies depending on the
    # test data) using a lambda function for convenience.
    def opt_FNN(w): return loss(y_train, F_FNN(x_train, w))
    wFNN = minimize(opt_FNN, np.random.randn(D)).x

    def opt_CNN(w): return loss(y_train, F_CNN(x_train, w))
    wCNN = minimize(opt_CNN, np.random.randn(FILTER_SIZE)).x

    # print(wCNN)
    # print(W0)
    # exit()

    # generating test data
    x_test, _ = datagen(N_TEST)
    y_test = F_CNN(x_test, W0)
    cnn_pred = F_CNN(x_test, wCNN)
    fnn_pred = F_FNN(x_test, wFNN)
    print('n={0} samples'.format(n))
    FNN_loss = loss(y_test, fnn_pred)
    CNN_loss = loss(y_test, cnn_pred)
    # print('FNN loss:{:.2e}'.format(FNN_loss))
    # print('CNN loss:{:.2e}'.format(CNN_loss))
    cnn_err.append(CNN_loss)
    fnn_err.append(FNN_loss)
    # print("sqrt(m/n) = {0:.2e} vs {1:.2e} = CNN_loss/FNN_loss".format(
    #     np.sqrt(FILTER_SIZE/D), CNN_loss/FNN_loss
    # ))
    # exit()

print("sqrt(1/500 / 1/2000) = 2 ~ {0:.2f} = cnn_err(500)/cnn_err(2000)".format(
    cnn_err[0]/cnn_err[-1]))
