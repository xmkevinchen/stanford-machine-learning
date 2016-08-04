import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import re
import scipy.optimize as sop
from ex8 import estimate_gaussian, multivariate_gaussian, select_threshold

def pack_parameters(t1, t2):
    return np.concatenate((t1.reshape(-1), t2.reshape(-1)))

def unpack_parameters(thetas, num_users, num_movies, num_features):
    X = thetas[0:num_movies * num_features].reshape((num_movies, num_features))
    Theta = thetas[num_movies * num_features:].reshape((num_users, num_features))

    return X, Theta

def cofi_cost(params, Y, R, num_users, num_movies, num_features, reg_lambda):

    X, Theta = unpack_parameters(params, num_users, num_movies, num_features)

    error = (X.dot(Theta.T) - Y) * R
    J = np.sum(error.reshape(-1, 1) ** 2) / 2


    J = J + (np.sum(Theta.reshape(-1, 1) ** 2)  + np.sum(X.reshape(-1, 1) ** 2)) * reg_lambda / 2

    return J

def cofi_cost_derivative(params, Y, R, num_users, num_movies, num_features, reg_lambda):
    X, Theta = unpack_parameters(params, num_users, num_movies, num_features)

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    error = (X.dot(Theta.T) - Y) * R

    X_grad = error.dot(Theta)
    Theta_grad = error.T.dot(X)

    X_grad = X_grad + reg_lambda * X
    Theta_grad = Theta_grad + reg_lambda * Theta

    grad = pack_parameters(X_grad, Theta_grad)

    return grad

def check_cost_function(reg_lambda = 0):
    '''
    Creates a collaborative filering problem to check your cost function and gradients,
    it will output the analytical gradients produced by your code and the numerical gradients.
    These two gradient computations should result in very similar values.
    '''

    X_t = np.random.rand(4,3)
    Theta_t = np.random.rand(5, 3)

    Y = X_t @ Theta_t.T
    m, n = Y.shape
    Y[np.where(np.random.rand(m,n) > 0.5)] = 0
    R = np.zeros(Y.shape).astype(int)
    R[np.where(Y != 0)] = 1

    m, n = X_t.shape
    X = np.random.randn(m, n)
    m, n = Theta_t.shape
    Theta = np.random.randn(m, n)
    num_movies, num_users = Y.shape
    num_features = Theta_t.shape[1]

    cofi_cost_func = lambda t: cofi_cost(t, Y, R, num_users, num_movies, num_features, reg_lambda)
    numgrad = compute_numerical_gradient(cofi_cost_func, pack_parameters(X, Theta))
    params = pack_parameters(X, Theta)

    cost = cofi_cost(params, Y, R, num_users, num_movies, num_features, reg_lambda)
    grad = cofi_cost_derivative(params, Y, R, num_users, num_movies, num_features, reg_lambda)

    print(np.array_str(np.array((numgrad, grad))))
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print("If your backpropagation implementation is correct, then\n"
          "the relative difference will be small (less than 1e-9).\n"
          "\nRelative Difference: {}".format(diff))


def compute_numerical_gradient(cost_func, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    epsilon = 1e-4
    for p in range(0, theta.size):
        perturb[p] = epsilon
        loss1 = cost_func(theta - perturb)
        loss2 = cost_func(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2 * epsilon)
        perturb[p] = 0

    return numgrad

def load_movie_list():
    f = open('movie_ids.txt', encoding='ISO-8859-1')

    movie_list = [re.split('\d+\s', line)[-1].strip() for line in f.readlines()]
    return movie_list

def normalize_ratings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)

    for i in range(m):
        idx = np.where(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean



if __name__ == '__main__':

    '''
    =============== Part 1: Loading movie ratings dataset ================
    You will start by loading the movie ratings dataset to understand the
    structure of the data.

    '''

    data = sio.loadmat('ex8_movies.mat')
    # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
    Y = data['Y']
    # R is 1682x943 matrix, where R[i, j] = 1 if and only if user j gave a rating to movie i
    R = data['R']

    print("Average rating for movie 1 (Toy Story): {:.2f} / 5".format(np.mean(Y[0, np.where(R[0, :] == 1)])))

    plt.imshow(Y)
    plt.ylabel('Movies')
    plt.xlabel('Users')

    '''
    ============ Part 2: Collaborative Filtering Cost Function ===========
    %  You will now implement the cost function for collaborative filtering.
    %  To help you debug your cost function, we have included set of weights
    %  that we trained on that. Specifically, you should complete the code in
    %  cofiCostFunc.m to return J.
    '''

    # Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    data = sio.loadmat('ex8_movieParams.mat')
    X = data['X']
    Theta = data['Theta']
    num_users = data['num_users']
    num_movies = data['num_movies']
    num_features = data['num_features']

    # Reduce the data set size so that this runs faster
    num_users = 4
    num_movies = 5
    num_features = 3

    X = X[0:num_movies, 0:num_features]
    Theta = Theta[0:num_users, 0:num_features]
    Y = Y[0:num_movies, 0:num_users]
    R = R[0:num_movies, 0:num_users]

    # Evaluate cost function
    J = cofi_cost(pack_parameters(X, Theta), Y, R, num_users, num_movies, num_features, 0)
    print("Cost at loaded paramaters: {:.2f}\n(this value should be about 22.22)".format(J))


    '''
    ============== Part 3: Collaborative Filtering Gradient ==============
    %  Once your cost function matches up with ours, you should now implement
    %  the collaborative filtering gradient function. Specifically, you should
    %  complete the code in cofiCostFunc.m to return the grad argument.
    %
    '''
    check_cost_function()

    '''
    ========= Part 4: Collaborative Filtering Cost Regularization ========
    %  Now, you should implement regularization for the cost function for
    %  collaborative filtering. You can implement it by adding the cost of
    %  regularization to the original cost computation.
    %
    '''
    J = cofi_cost(pack_parameters(X, Theta), Y, R, num_users, num_movies, num_features, 1.5)
    print("\nCost at loaded paramaters (lambda = 1.5): {:.2f}\n(this value should be about 31.34)".format(J))


    '''
    ======= Part 5: Collaborative Filtering Gradient Regularization ======
    %  Once your cost matches up with ours, you should proceed to implement
    %  regularization for the gradient.
    %
    '''

    print("Checking Gradients (with regularization) ...")
    check_cost_function(1.5)

    '''
    %% ============== Part 6: Entering ratings for a new user ===============
    %  Before we will train the collaborative filtering model, we will first
    %  add ratings that correspond to a new user that we just observed. This
    %  part of the code will also allow you to put in your own ratings for the
    %  movies in our dataset!
    %
    '''
    movie_list = load_movie_list()
    ratings = np.zeros(len(movie_list))

    ratings[0] = 4;
    ratings[97] = 2;
    ratings[6] = 3;
    ratings[11]= 5;
    ratings[53] = 4;
    ratings[63]= 5;
    ratings[65]= 3;
    ratings[68] = 5;
    ratings[182] = 4;
    ratings[225] = 5;
    ratings[354]= 5;

    print("\nNew user rating:")
    for idx in range(len(ratings)):
        if ratings[idx] > 0:
            print("Rated {:.0f} for {}".format(ratings[idx], movie_list[idx]))

    '''
    %% ================== Part 7: Learning Movie Ratings ====================
    %  Now, you will train the collaborative filtering model on a movie rating
    %  dataset of 1682 movies and 943 users
    %
    '''

    data = sio.loadmat('ex8_movies.mat')
    Y = data['Y']
    R = data['R']

    Y = np.c_[ratings, Y]
    R = np.c_[(ratings != 0).astype(int), R]

    Ynorm, Ymean = normalize_ratings(Y, R)

    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10

    # Set Initial parameters (Theta, X)
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.rand(num_users, num_features)

    initial_parameters = pack_parameters(X, Theta)

    reg_lambda = 10
    result = sop.minimize(cofi_cost,
                          x0=initial_parameters,
                          args=(Y, R, num_users, num_movies, num_features, reg_lambda),
                          method='CG',
                          jac=cofi_cost_derivative,
                          options={"disp": True, "maxiter": 100})

    X, Theta = unpack_parameters(result.x, num_users, num_movies, num_features)

    print("Recommender system learning completed.")

    p = X @ Theta.T
    predictions = p[:, 0] + Ymean

    idx = np.argsort(predictions)[::-1]
    print("\nTop recommendations for you")
    for i in range(10):
        j = idx[i]
        print("Predicting rate {:.1f} for movie {}".format(predictions[j], movie_list[j]))

    print("\n\nOriginal ratings provided:")
    for i in range(len(ratings)):
        if ratings[i] > 0:
            print("Rated {:.0f} for {}".format(ratings[i], movie_list[i]))
