"""
Test code for segmentation module in 8DC00 course
"""

# Imports

import numpy as np
import segmentation as seg
import segmentation_util as util
import matplotlib.pyplot as plt
from scipy import ndimage, stats, spatial
import scipy
from sklearn.neighbors import KNeighborsClassifier
import timeit
from IPython.display import display, clear_output

plt.rcParams['image.cmap'] = 'gray'


# SECTION 1. Segmentation in feature space

def scatter_data_test(showFigs=True):
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    X1 = I.flatten().T
    X1 = X1.reshape(-1, 1)
    GT = plt.imread('../data/dataset_brains/1_1_gt.tif')
    gt_mask = GT > 0
    Y = gt_mask.flatten()  # labels

    I_blurred = ndimage.gaussian_filter(I, sigma=2)
    X2 = I_blurred.flatten().T
    X2 = X2.reshape(-1, 1)

    X_data = np.concatenate((X1, X2), axis=1)

    features = ('T1 intensity', 'T1 gauss 2')  # Keep track of features you added
    if showFigs:
        ax = util.scatter_data(X_data, Y, 0, 1)
        return X_data, Y

    return X_data, Y


def scatter_t2_test(showFigs=True):
    I1 = plt.imread('../data/dataset_brains/1_1_t1.tif')
    X1 = I1.flatten().T
    X1 = X1.reshape(-1, 1)

    I2 = plt.imread('../data/dataset_brains/1_1_t2.tif')
    X2 = I2.flatten().T
    X2 = X2.reshape(-1, 1)

    GT = plt.imread('../data/dataset_brains/1_1_gt.tif')
    gt_mask = GT > 0
    Y = gt_mask.flatten()  # labels

    I1_blurred = ndimage.gaussian_filter(I1, sigma=4)
    X12 = I1_blurred.flatten().T
    X12 = X12.reshape(-1, 1)

    X_data = np.concatenate((X1, X12), axis=1)
    features = ('T1 intensity', 'T1 gauss 2')  # Keep track of features you added
    if showFigs:
        util.scatter_data(X_data, Y, 0, 1)

    return X_data, Y


def extract_coordinate_feature_test():
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    c, coord_im = seg.extract_coordinate_feature(I)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121)
    ax1.imshow(I)
    ax2 = fig.add_subplot(122)
    ax2.imshow(coord_im)


def feature_stats_test():
    X, Y = scatter_data_test(showFigs=False)
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    c, coord_im = seg.extract_coordinate_feature(I)
    X_data = np.concatenate((X, c), axis=1)

    print(np.mean(X_data, axis=0))
    print(np.std(X_data, axis=0))


def normalized_stats_test():
    X, Y = scatter_data_test(showFigs=False)
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    c, coord_im = seg.extract_coordinate_feature(I)
    X_data = np.concatenate((X, c), axis=1)

    X_data_norm, _ = seg.normalize_data(X_data)
    print(np.mean(X_data_norm, axis=0))
    print(np.std(X_data_norm, axis=0))


def distance_test():
    X, Y = seg.generate_gaussian_data()
    D = scipy.spatial.distance.cdist(X, X, metric='euclidean')
    ax = plt.imshow(D)


def small_samples_distance_test():
    X, Y = seg.generate_gaussian_data(50)
    C = np.array([[0, 0], [1, 1]])
    D = scipy.spatial.distance.cdist(X, C, metric='euclidean')

    return X, Y, C, D


def minimum_distance_test(X, Y, C, D):
    plt.plot(X[:, 0], X[:, 1], '.r')
    plt.plot(C, '.b')
    min_index = np.argmin(D, axis=1)
    min_dist = np.min(D, axis=1)
    n0 = np.sum(min_index == 0)
    n1 = np.sum(min_index == 1)
    print(n0, n1)


def distance_classification_test():
    train_data, train_labels = seg.generate_gaussian_data(2)
    test_data, test_labels = seg.generate_gaussian_data(1)

    D = scipy.spatial.distance.cdist(test_data, train_data, metric='euclidean')
    test_labels = train_labels[np.argmin(D, axis=1)]

    print(test_labels)


def funX(X):
    return lambda w: seg.cost_kmeans(X, w)


def kmeans_demo():
    # Define some data and parameters
    n = 100
    X1 = np.random.randn(n, 2)
    X2 = np.random.randn(n, 2) + 5
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((np.zeros((n, 1)), np.ones((n, 1))), axis=0)
    #     ax1 = util.scatter_data(X,Y,0,1)
    N, M = X.shape

    # Define number of clusters we want
    clusters = 2;

    # the learning rate
    mu = 1;

    # iterations
    num_iter = 100

    # Cost function used by k-Means
    # fun = lambda w: seg.cost_kmeans(X,w)
    fun = funX(X)

    ## Algorithm
    # Initialize cluster centers
    idx = np.random.randint(N, size=clusters)
    initial_w = X[idx, :]
    w_draw = initial_w
    print(w_draw)

    # Reshape into vector (needed by ngradient)
    w_vector = initial_w.reshape(clusters * M, 1)

    # Vector to store cost
    xx = np.linspace(1, num_iter, num_iter)
    kmeans_cost = np.empty(*xx.shape)
    kmeans_cost[:] = np.nan

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    im1 = ax1.scatter(X[:n, 0], X[:n, 1], label='X-class0')
    im2 = ax1.scatter(X[n:, 0], X[n:, 1], label='X-class1')
    line1, = ax1.plot(w_draw[:, 0], w_draw[:, 1], "or", markersize=5, label='W-vector')
    # im3  = ax1.scatter(w_draw[:,0], w_draw[:,1])
    ax1.grid()

    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 10))

    text_str = 'k={}, g={:.2f}\ncost={:.2f}'.format(0, 0, 0)

    txt2 = ax2.text(0.3, 0.95, text_str, bbox={'facecolor': 'green', 'alpha': 0.4, 'pad': 10},
                    transform=ax2.transAxes)

    #     xx = xx.reshape(1,-1)
    line2, = ax2.plot(xx, kmeans_cost, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost')
    ax2.grid()

    for k in np.arange(num_iter):
        # gradient ascent
        g = util.ngradient(fun, w_vector)
        w_vector = w_vector - mu * g
        # calculate cost for plotting
        kmeans_cost[k] = fun(w_vector)
        text_str = 'k={}, cost={:.2f}'.format(k, kmeans_cost[k])
        txt2.set_text(text_str)
        # plot
        line2.set_ydata(kmeans_cost)
        w_draw_new = w_vector.reshape(clusters, M)
        line1.set_data(w_draw_new[:, 0], w_draw_new[:, 1])
        display(fig)
        clear_output(wait=True)
        plt.pause(.005)

    return kmeans_cost


def kmeans_clustering_test():
    I = plt.imread('../data/dataset_brains/1_1_t1.tif')
    X, _ = seg.normalize_data(I)
    labels = seg.kmeans_clustering(X)
    # plt.imshow(labels)
    print(labels)


def nn_classifier_test_samples():
    train_data, train_labels = seg.generate_gaussian_data(20)
    test_data, test_labels = seg.generate_gaussian_data(10)
    predicted_labels = seg.nn_classifier(train_data, train_labels, test_data)

    # predicted_labels = predicted_labels.astype(bool)
    # test_labels = test_labels.astype(bool)
    err = util.classification_error(test_labels, predicted_labels)

    print('True labels:\n{}'.format(test_labels))
    print('Predicted labels:\n{}'.format(predicted_labels))
    print('Error:\n{}'.format(err))


def generate_train_test(N, task):
    # generates a training and a test set with the same
    # data distribution from two possibilities: easy dataset with low class
    # overlap, or hard dataset with high class overlap
    #
    # Input:
    #
    # N             - Number of samples per classs
    # task          - String, either 'easy' or 'hard'

    if task == 'easy':
        mu1 = [0, 0]
        mu2 = [4, 2]
        sigma1 = [[1, 0], [0, 1]]
        sigma2 = [[1, -1], [-1, 3]]

    if task == 'hard':
        mu1 = [0, 0]
        mu2 = [1, 1]
        sigma1 = [[3, 0], [0, 2]]
        sigma2 = [[2, 0], [0, 3]]

    trainX, trainY = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)
    testX, testY = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    return trainX, trainY, testX, testY


def easy_hard_data_classifier_test():
    train_data_easy, train_label_easy, test_data_easy, test_label_easy = generate_train_test(50, 'easy')
    train_data_hard, train_label_hard, test_data_hard, test_label_hard = generate_train_test(50, 'hard')

    predicted_labels_easy = seg.nn_classifier(train_data_easy, train_label_easy, test_data_easy)
    predicted_labels_hard = seg.nn_classifier(train_data_hard, train_label_hard, test_data_hard)

    err_easy = util.classification_error(test_label_easy, predicted_labels_easy)
    err_hard = util.classification_error(test_label_hard, predicted_labels_hard)

    print("easy: {0}, hard: {1}".format(err_easy, err_hard))


def nn_classifier_test_brains(testDice=False):
    # Subject 1, slice 1 is the train data
    X, Y, feature_labels_train = util.create_dataset(1, 1, 'brain')
    N = 1000
    ix = np.random.randint(len(X), size=N)
    train_data = X[ix, :]
    train_labels = Y[ix, :]
    # Subject 3, slice 1 is the test data
    test_data, test_labels, feature_labels_test = util.create_dataset(3, 1, 'brain')

    predicted_labels = seg.nn_classifier(train_data, train_labels, test_data)
    predicted_labels = predicted_labels.astype(bool)
    test_labels = test_labels.astype(bool)
    err = util.classification_error(test_labels, predicted_labels)
    print('Error:\n{}'.format(err))

    if testDice:
        dice = util.dice_overlap(test_labels, predicted_labels)
        print('Dice coefficient:\n{}'.format(dice))
    else:
        I = plt.imread('../data/dataset_brains/3_1_t1.tif')
        GT = plt.imread('../data/dataset_brains/3_1_gt.tif')
        gt_mask = GT > 0
        gt_labels = gt_mask.flatten()  # labels
        predicted_mask = predicted_labels.reshape(I.shape)
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax1.imshow(I)
        ax2 = fig.add_subplot(132)
        ax2.imshow(predicted_mask)
        ax3 = fig.add_subplot(133)
        ax3.imshow(gt_mask)


def knn_curve():
    # Load training and test data
    train_data, train_labels, train_feature_labels = util.create_dataset(1, 1, 'brain')
    test_data, test_labels, test_feature_labels = util.create_dataset(2, 1, 'brain')
    # Normalize data
    train_data, test_data = seg.normalize_data(train_data, test_data)

    # Define parameters
    num_iter = 3
    train_size = 100
    k = np.array([1, 3, 5, 9, 15, 25, 100])
    # k = np.array([1, 5, 9])

    # Store errors
    test_error = np.empty([len(k), num_iter])
    test_error[:] = np.nan
    dice = np.empty([len(k), num_iter])
    dice[:] = np.nan

    ## Train and test with different values

    for i in np.arange(len(k)):
        for j in np.arange(num_iter):
            print('k = {}, iter = {}'.format(k[i], j))
            # Subsample training set
            ix = np.random.randint(len(train_data), size=train_size)
            subset_train_data = train_data[ix, :]
            subset_train_labels = train_labels[ix, :]

            predicted_test_labels = seg.knn_classifier(subset_train_data, subset_train_labels, test_data, k[i])

            # #Train classifier
            # neigh = KNeighborsClassifier(n_neighbors=k[i])
            # neigh.fit(subset_train_data, subset_train_labels)
            # #Evaluate
            # predicted_test_labels = neigh.predict(test_data)

            test_error[i, j] = util.classification_error(test_labels, predicted_test_labels)
            dice[i, j] = util.dice_overlap(test_labels, predicted_test_labels)

    ## Display results
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    p1 = ax1.plot(k, np.mean(test_error, 1), 'r', label='error')
    p2 = ax1.plot(k, np.mean(dice, 1), 'k', label='dice')
    ax1.set_xlabel('k')
    ax1.set_ylabel('error')
    ax1.grid()
    ax1.legend()


# SECTION 2. Generalization and overfitting

def learning_curve():
    # Load training and test data
    # train_data, train_labels = seg.generate_gaussian_data(1000)
    train_data, train_labels, _ = util.create_dataset(1, 1, 'brain')
    # test_data, test_labels = seg.generate_gaussian_data(1000)
    test_data, test_labels, _ = util.create_dataset(2, 1, 'brain')
    train_data, test_data = seg.normalize_data(train_data, test_data)

    # Define parameters
    train_sizes = np.logspace(0.1, 3.0, num=15).astype(int)
    k = 1
    num_iter = 3  # How often to repeat the experiment

    # Store errors
    test_error = np.empty([len(train_sizes), num_iter])
    test_error[:] = np.nan
    test_dice = np.empty([len(train_sizes), num_iter])
    test_dice[:] = np.nan

    train_error = np.empty([len(train_sizes), num_iter])
    train_error[:] = np.nan
    train_dice = np.empty([len(train_sizes), num_iter])
    train_dice[:] = np.nan

    ## Train and test with different values
    for i in np.arange(len(train_sizes)):
        for j in np.arange(num_iter):
            print('train_size = {}, iter = {}'.format(train_sizes[i], j))
            # Subsample training set
            ix = np.random.randint(len(train_data), size=train_sizes[i])
            subset_train_data = train_data[ix, :]
            subset_train_labels = train_labels[ix, :]

            # Train classifier
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(subset_train_data, subset_train_labels.ravel())
            # Evaluate
            predicted_test_labels = neigh.predict(test_data)

            test_labels = test_labels.astype(bool)
            predicted_test_labels = predicted_test_labels.astype(bool)

            test_error[i, j] = util.classification_error(test_labels, predicted_test_labels)
            test_dice[i, j] = util.dice_overlap(test_labels, predicted_test_labels)

            predicted_train_labels = neigh.predict(train_data).astype(bool)
            train_labels_bool = train_labels.astype(bool)

            train_error[i, j] = util.classification_error(train_labels_bool, predicted_train_labels)
            train_dice[i, j] = util.dice_overlap(train_labels_bool, predicted_train_labels)

    ## Display results
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    x = np.log(train_sizes)
    ticks = list(x)
    tick_lbls = [str(i) for i in train_sizes]

    y_test = np.mean(test_error, 1)
    y_train = np.mean(train_error, 1)

    yerr_test = np.std(test_error, 1)
    yerr_train = np.std(train_error, 1)

    p1 = ax1.errorbar(x, y_test, yerr=yerr_test, label='Test error')
    p2 = ax2.errorbar(x, y_train, yerr=yerr_train, label='Train error')

    ax1.set_xlabel('Number of training samples (k)')
    ax1.set_ylabel('error')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_lbls)
    ax1.grid()
    ax1.legend()

    ax2.set_xlabel('Number of training samples (k)')
    ax2.set_ylabel('error')
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_lbls)
    ax2.grid()
    ax2.legend()


def feature_curve(use_random=False):
    # Load training and test data
    train_data, train_labels, train_feature_labels = util.create_dataset(1, 1, 'brain')
    test_data, test_labels, test_feature_labels = util.create_dataset(2, 1, 'brain')

    if use_random:
        train_data = np.random.randn(train_data.shape[0], train_data.shape[1])

    # Normalize data
    train_data, test_data = seg.normalize_data(train_data, test_data)

    # Define parameters
    feature_sizes = np.arange(train_data.shape[1]) + 1
    train_size = 10
    k = 3
    num_iter = 5

    # Store errors
    test_error = np.empty([len(feature_sizes), num_iter])
    test_error[:] = np.nan
    train_error = np.empty([len(feature_sizes), num_iter])
    train_error[:] = np.nan

    # Train and test with different sizes
    for i in np.arange(len(feature_sizes)):
        for j in np.arange(num_iter):
            print('feature size = {}, iter = {}'.format(feature_sizes[i], j))
            start_time = timeit.default_timer()
            # Subsample training set
            ix = np.random.randint(len(train_data), size=train_size)
            subset_train_data = train_data[ix, :]
            subset_train_labels = train_labels[ix, :]

            # Train classifier
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(subset_train_data[:, :feature_sizes[i]], subset_train_labels.ravel())
            # Evaluate
            predicted_test_labels = neigh.predict(test_data[:, :feature_sizes[i]])
            predicted_train_labels = neigh.predict(subset_train_data[:, :feature_sizes[i]])

            test_error[i, j] = util.classification_error(test_labels, predicted_test_labels)
            train_error[i, j] = util.classification_error(subset_train_labels, predicted_train_labels)

            # Timer log
            elapsed = timeit.default_timer() - start_time
            # print('elapsed time = {}'.format(elapsed))

    ## Display results
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    x = feature_sizes
    y_test = np.mean(test_error, 1)
    yerr_test = np.std(test_error, 1)
    p1 = ax1.errorbar(x, y_test, yerr=yerr_test, label='Test error')

    ax1.set_xlabel('Number of features')
    ax1.set_ylabel('Error')
    ax1.grid()
    ax1.legend()


def high_dimensions_demo():
    # Base figure
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)

    # Generate data from 2D Gaussian distribution
    X1 = np.random.randn(100, 2)
    mn1, mx1, mns1 = high_dimensions_output(X1, ax1, 20)
    print('2D Gaussian distribution')
    print('Mean = {}, Max = {}, Mean nn = {}'.format(mn1, mx1, mns1))

    # Generate data from 1000D Gaussian distribution
    X2 = np.random.randn(100, 1000)
    mn2, mx2, mns2 = high_dimensions_output(X2, ax2, 20)
    print('1000D Gaussian distribution')
    print('Mean = {}, Max = {}, Mean nn = {}'.format(mn2, mx2, mns2))

    ## More surprising properties of high dimensions
    n = 10
    k = 0.01
    frac = np.arange(n)
    for i in np.arange(n):
        frac[i] = k ** i

    ax3 = fig.add_subplot(133)
    ax3.plot(np.arange(n), frac)
    ax3.set_xlabel('dimensions')
    ax3.set_ylabel('fraction to travel per dimension')


def high_dimensions_output(X, ax, n_bins=20):
    # Calculate distances
    D = scipy.spatial.distance.cdist(X, X, metric='euclidean')

    # Plot histogram
    ax.hist(D.flatten(), bins=n_bins)

    mn = np.mean(D)
    mx = np.max(D)

    mns = 0
    for n, i in enumerate(D):
        mns += np.min(np.delete(i, n, None), axis=None)
    mns = mns / len(X)

    return mn, mx, mns


def covariance_matrix_test():
    N = 100
    mu1 = [0, 0]
    mu2 = [0, 0]
    sigma1 = [[3, 1], [1, 1]]
    sigma2 = [[3, 1], [1, 1]]
    X, Y = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    sigma = np.cov(X.T)
    mu = np.mean(X, axis=0)

    return X, Y, sigma


def eigen_vecval_test(sigma):
    w, v = np.linalg.eig(sigma)

    ix = np.argsort(w)[::-1]
    w = w[ix]
    v = v[:, ix]

    return v, w


def rotate_using_eigenvectors_test(X, Y, v):
    return v.T.dot(X.T).T


def test_mypca():
    # Generates some toy data in 2D, computes PCA, and plots both datasets
    N = 100
    mu1 = [0, 0]
    mu2 = [2, 0]
    sigma1 = [[2, 1], [1, 1]]
    sigma2 = [[2, 1], [1, 1]]

    XG, YG = seg.generate_gaussian_data(N, mu1, mu2, sigma1, sigma2)

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)

    util.scatter_data(XG, YG, ax=ax1)
    sigma = np.cov(XG, rowvar=False)
    w, v = np.linalg.eig(sigma)
    ax1.plot([0, v[0, 0]], [0, v[1, 0]], c='g', linewidth=3, label='Eigenvector1')
    ax1.plot([0, v[0, 1]], [0, v[1, 1]], c='k', linewidth=3, label='Eigenvector2')
    ax1.set_title('Original data')
    ax_settings(ax1)

    ax2 = fig.add_subplot(122)
    X_pca, v, w, fraction_variance = seg.mypca(XG)
    util.scatter_data(X_pca, YG, ax=ax2)
    sigma2 = np.cov(X_pca, rowvar=False)
    w2, v2 = np.linalg.eig(sigma2)
    ax2.plot([0, v2[0, 0]], [0, v2[1, 0]], c='g', linewidth=3, label='Eigenvector1')
    ax2.plot([0, v2[0, 1]], [0, v2[1, 1]], c='k', linewidth=3, label='Eigenvector2')
    ax2.set_title('My PCA')
    ax_settings(ax2)

    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  bbox_transform=plt.gcf().transFigure, ncol=4)

    print(fraction_variance)


def ax_settings(ax):
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()


# SECTION 3. Atlases and active shapes

def segmentation_combined_atlas_test():
    task = 'brain'
    n = 5
    all_subjects = np.arange(n)
    train_slice = 1
    tmp_data, tmp_labels, tmp_feature_labels = util.create_dataset(1, train_slice, task)
    all_data_matrix = np.empty((tmp_data.shape[0], tmp_data.shape[1], n))
    all_labels_matrix = np.empty((tmp_labels.shape[0], n))

    # Load datasets once
    for i in all_subjects:
        train_data, train_labels, train_feature_labels = util.create_dataset(i + 1, train_slice, task)
        all_data_matrix[:, :, i] = train_data
        all_labels_matrix[:, i] = train_labels.ravel()

    # Combine labels of training images:
    predicted_labels = seg.segmentation_combined_atlas(all_labels_matrix, combining='mode')

    # Convert combined label into mask image:
    predicted_mask = predicted_labels.reshape(240, 240)

    # Convert true label into mask image:
    true_mask = all_labels_matrix[:, 4].reshape(240, 240)

    plt.imshow(predicted_mask + true_mask)

    err = util.classification_error(true_mask, predicted_mask)
    dice = util.dice_overlap(true_mask, predicted_mask)

    print("error: {0}, dice: {1}".format(err, dice))


def segmentation_combined_atlas_minmax_test():
    task = 'brain'
    n = 5
    all_subjects = np.arange(n)
    train_slice = 1
    tmp_data, tmp_labels, tmp_feature_labels = util.create_dataset(1, train_slice, task)
    all_data_matrix = np.empty((tmp_data.shape[0], tmp_data.shape[1], n))
    all_labels_matrix = np.empty((tmp_labels.shape[0], n))

    # Load datasets once
    for i in all_subjects:
        train_data, train_labels, train_feature_labels = util.create_dataset(i + 1, train_slice, task)
        all_data_matrix[:, :, i] = train_data
        all_labels_matrix[:, i] = train_labels.ravel()

    predicted_labels = seg.segmentation_combined_knn(all_data_matrix[:, :, :n - 2], all_labels_matrix[:, :n - 2],
                                                     all_data_matrix[:, :, n - 1])

    # predicted_labels_max = seg.segmentation_combined_atlas(all_labels_matrix, combining='max')
    # predicted_labels_min = seg.segmentation_combined_atlas(all_labels_matrix, combining='min')

    test_labels = all_labels_matrix[:, n - 1].astype(bool)

    # print('Combining method = min:')
    # err = util.classification_error(test_labels, predicted_labels_min)
    # print('Error:\n{}'.format(err))
    # dice = util.dice_overlap(test_labels, predicted_labels_min)
    # print('Dice coefficient:\n{}'.format(dice))

    print()

    print('Combining knn')
    err = util.classification_error(test_labels, predicted_labels)
    print('Error:\n{}'.format(err))
    dice = util.dice_overlap(test_labels, predicted_labels)
    print('Dice coefficient:\n{}'.format(dice))

    plt.imshow(predicted_labels.reshape(240, 240))
    plt.show()
    plt.imshow(test_labels.reshape(240, 240))
