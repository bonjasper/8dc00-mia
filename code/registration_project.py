"""
Registration project code.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output


def point_based_affine_registration(I, Im):
    # let the user selecte points
    X, Xm = util.my_cpselect(I, Im)

    # find affine transformation matrix
    T, reg_error = reg.ls_affine(util.c2h(X), util.c2h(Xm))

    # transform the moving image according to T
    Im = plt.imread(Im)
    It, _ = reg.image_transform(Im, T)

    return It, reg_error


def intensity_based_registration_demo(I, Im, mu=0.0005, num_iter=100, h=1e-3, x=np.array([0., 1., 1., 0., 0., 0., 0.]), type="affine", sim_meas="mi"):
    # read the fixed and moving images
    # change these in order to read different images
    # I = plt.imread('../data/image_data/1_1_t1.tif')
    # Im = plt.imread('../data/image_data/1_1_t2.tif')

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    # x = np.array([0., 1., 1., 0., 0., 0., 0.])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation

    assert type.lower() in ["affine", "rigid"], "error: unknown type"
    assert sim_meas.lower() in ["mi", "cc"], "error: unknown similarity measure"

    if type == "affine":
        if sim_meas == "mi":
            fun = lambda x: reg.affine_mi(I, Im, x)
        else:
            fun = lambda x: reg.affine_corr(I, Im, x)
    else:
        if sim_meas == "cc":
            fun = lambda x: reg.rigid_corr(I, Im, x)
        else:
            ModuleNotFoundError("no functionality for type=rigid and sim_meas=mi")

    # the learning rate
    # mu = 0.0005

    # number of iterations
    # num_iter = 100

    iterations = np.arange(1, num_iter + 1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14, 6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(Im, alpha=0.5)
    # parameters
    txt = ax1.text(0.3, 0.95,
                   np.array2string(x, precision=5, floatmode='fixed'),
                   bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
                   transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 2))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()


    path = []
    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):
        # gradient ascent
        g = reg.ngradient(fun, x, h=h)
        x += g * mu
        path.append(x.copy())

        # for visualization of the result
        S, Im_t, _ = fun(x)

        clear_output(wait=True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        if k % int(num_iter / 10) == 0:
            print("biep, {:4.0%}...".format(k / num_iter))

    print("helemaal klaar dr mee!")

    fig.show()
    return path
