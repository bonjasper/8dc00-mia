"""
Utility functions for registration.
"""

import numpy as np
from cpselect.cpselect import cpselect


def test_object(centered=True):
    # Generate an F-like test object.
    # Input:
    # centered - set the object centroid to the origin
    # Output:
    # X - coordinates of the test object

    X = np.array([[4, 4, 4.5, 4.5, 6, 6, 4.5, 4.5, 7, 7, 4], [10, 4, 4, 7, 7, 7.5, 7.5, 9.5, 9.5, 10, 10]])

    if centered:
        X[0, :] = X[0, :] - np.mean(X[0, :])
        X[1, :] = X[1, :] - np.mean(X[1, :])

    return X


def c2h(X):
    # Convert cartesian to homogeneous coordinates.
    # Input:
    # X - cartesian coordinates
    # Output:
    # Xh - homogeneous coordinates

    n = np.ones([1, X.shape[1]])
    Xh = np.concatenate((X, n))

    return Xh


def t2h(T, Xt=[0, 0]):
    # Convert a 2D transformation matrix to homogeneous form.
    # Input:
    # T - 2D transformation matrix
    # Xt - 2D translation vector
    # Output:
    # Th - homogeneous transformation matrix

    Th = np.array([[T[0, 0], T[0, 1], Xt[0]], [T[1, 0], T[1, 1], Xt[1]], [0, 0, 1]])

    return Th


def plot_object(ax, X):
    # Plot 2D object.
    #
    # Input:
    # X - coordinates of the shape

    ax.plot(X[0, :], X[1, :], linewidth=2);


def my_cpselect(I_path, Im_path):
    # Wrapper around cpselect that returns the point coordinates
    # in the expected format (coordinates in rows).
    # Input:
    # I - fixed image
    # Im - moving image
    # Output:
    # X - control points in the fixed image
    # Xm - control points in the moving image

    # get the input_points and check if the user specified enough points

    input_points = cpselect(I_path, Im_path)
    # input_points = [{'point_id': 1, 'img1_x': 125.05849883303026, 'img1_y': 108.66886798820963, 'img2_x': 122.08820593487474, 'img2_y': 122.60132079292578}, {'point_id': 2, 'img1_x': 143.85110959287996, 'img1_y': 171.85092140494567, 'img2_x': 136.02065873959083, 'img2_y': 185.78337420966182}, {'point_id': 3, 'img1_x': 158.10757292793835, 'img1_y': 203.6039533784848, 'img2_x': 147.0370167712269, 'img2_y': 220.77651148662332}, {'point_id': 4, 'img1_x': 160.375646640334, 'img1_y': 104.78074162410277, 'img2_x': 157.40535374217842, 'img2_y': 121.62928920189907}, {'point_id': 5, 'img1_x': 160.375646640334, 'img1_y': 169.58284769255002, 'img2_x': 152.54519578704497, 'img2_y': 186.4313952703463}]
    if len(input_points) < 2: raise ValueError('Please select 2 or more points in the image')

    # initialize the array as a list and loop over input_points to extract the values adn add them as a row
    array = []
    for i in range(len(input_points)):
        array.append(list(input_points[i].values())[1:])

    # change from a list to an array, transpose and select the right rows for the output array
    array = np.array(array).transpose()
    X = array[:2, :]
    Xm = array[2:, :]

    return X, Xm
