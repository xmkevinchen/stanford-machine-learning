import numpy as np
import scipy.io as sio
from scipy.optimize import minimize

def forward_propagation(X, t1, t2):
    m = X.shape[0]

    ones = None
    if len(X.shape) == 1:
        ones = np.array(1).reshape(1,)
    else:
        ones = np.ones(m).reshape(m,1)

    a1 = np.hstack((ones, X))

    z2 = np.dot(a1, t1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((ones, a2))

    z3 = np.dot(a2, t2.T)
    H = sigmoid(z3)

    return a1, z2, a2, z3, H

def nn_cost_function(thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
    t1, t2 = unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)
    m = X.shape[0]

    Y = np.zeros((m, num_labels))
    for i in range(0, m):
        Y[i, np.asscalar(y[i]) - 1] = 1

    ones = None
    if len(X.shape) == 1:
        ones = np.array(1).reshape(1,)
    else:
        ones = np.ones(m).reshape(m,1)

    a1 = np.hstack((ones, X))

    _, _, _, _, H = forward_propagation(X, t1, t2)

    J = np.sum(np.sum(-Y * np.log(H) - (1 - Y) * np.log(1 - H))) / m

    if reg_lambda != 0:
        t1_reg = t1[:, 1:]
        t2_reg = t2[:, 1:]
        reg = (np.sum(t1_reg ** 2) + np.sum(t2_reg ** 2)) * reg_lambda / (2 * m)
        J = J + reg

    return J

def nn_cost_function_derivative(thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
    t1, t2 = unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

    m = X.shape[0]

    a1, z2, a2, z3, H = forward_propagation(X, t1, t2)

    Y = np.zeros((m, num_labels))
    for i in range(0, m):
        Y[i, np.asscalar(y[i]) - 1] = 1

    ones = np.ones((m, 1))
    d3 = H - Y
    d2 = np.dot(d3, t2) * sigmoid_gradient(np.hstack((ones, z2)))
    d2 = d2[:, 1:]

    theta1_grad = np.dot(d2.T, a1) / m + reg_lambda * np.hstack((np.zeros((t1.shape[0], 1)), t1[:, 1:])) / m
    theta2_grad = np.dot(d3.T, a2) / m + reg_lambda * np.hstack((np.zeros((t2.shape[0], 1)), t2[:, 1:])) / m

    grad = pack_thetas(theta1_grad, theta2_grad)

    return grad

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def pack_thetas(t1, t2):
    return np.concatenate((t1.reshape(-1), t2.reshape(-1)))

def unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels):
    t1_start = 0
    t1_end = hidden_layer_size * (input_layer_size + 1)
    t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, (input_layer_size + 1)))
    t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))

    return t1, t2

def rand_init(l_in, l_out):
    np.random.seed(1)
    return np.random.rand(l_out, l_in + 1) * 2 * 0.12 - 0.12

def predict(X, t1, t2):
    _, _, _, _, h = forward_propagation(X, t1, t2)
    pred = np.argmax(h, 1) + 1
    return pred.reshape(len(pred), 1)

def check_nn_gradients(reg_lambda=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 5
    m = 5

    t1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    t2 = debug_initialize_weights(num_labels, hidden_layer_size)
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, 6), 3)

    nn_params = pack_thetas(t1, t2)

    cost_func = lambda p: nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)
    cost = cost_func(nn_params)
    grad = nn_cost_function_derivative(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)
    numgrad = compute_numerical_gradient(cost_func, nn_params)
    diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)

    print(np.array_str(np.array((numgrad, grad))))

    return diff

def compute_numerical_gradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)

    e = 1e-4
    for p in range(0, theta.size):
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad



def debug_initialize_weights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.sin(np.arange(1,W.size+1)).reshape(W.shape) / 10
    return W

data = sio.loadmat('ex4data1.mat')
X = data['X']
y = data['y']


thetas = sio.loadmat('ex4weights.mat')
theta1 = thetas['Theta1']
theta2 = thetas['Theta2']

input_layer_size = X.shape[1]
hidden_layer_size = theta1.shape[0]
reg_lambda = 0
num_labels = len(np.unique(y))

# thetas = np.concatenate((theta1.reshape(-1), theta2.reshape(-1)))
# J = nn_cost_function(thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)
#
# print('Cost at parameters (loaded from ex4weights): %f' % J)

# g = sigmoid_gradient(np.array((1,-0.5, 0, 0.5, 1)))
# print('Sigmoid gradient evaluated at [1,-0.5, 0, 0.5, 1]:\n' + np.array_str(g))

theta1 = rand_init(input_layer_size, hidden_layer_size)
theta2 = rand_init(hidden_layer_size, num_labels)

thetas = pack_thetas(theta1, theta2)
# options = {'maxiter': 400, 'disp': True}
options = {'disp': True}

cost_func = lambda t: nn_cost_function(t, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)
gradient_func = lambda t: nn_cost_function_derivative(t, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)

result = minimize(fun=cost_func,
                x0=thetas,
                method='TNC',
                jac = gradient_func,
                options=options)

t1, t2 = unpack_thetas(result.x, input_layer_size, hidden_layer_size, num_labels)

pred = predict(X, t1, t2)
print('Training Set Accuracy: %f' % (np.mean(np.double(pred == y)) * 100))
