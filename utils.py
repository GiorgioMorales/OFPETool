import os
import sys
import cv2
import h5py
import gdal
import numpy as np
import pandas as pd
from itertools import cycle, islice


def readRaster(path):
    """Read georreferenced raster (one band per image) as numpy array"""
    ds = gdal.Open(path)
    img_array = ds.GetRasterBand(1).ReadAsArray()
    return img_array


def add_replace_column_CSV(path, column_name, column_data, coords, year=None, field=None):
    """Add or replace a column to a CSV file
    @param path: Path of the CSV file
    @param column_name: Name of the column you want to insert or replace.
    @param column_data: Vector of yield values.
    @param coords: Vector of (X, Y) coordinates corresponding to the yield vector.
    @param year: Specify year that will be modified.
    @param field: Specify field that will be modified.
    """
    # Read data
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    if column_name not in df.columns:
        df[column_name] = ""
    # Find correspondence between coordinates of each row and "coords"
    print("Saving predicted yield values to CSV file...")
    for c in range(len(coords)):
        if coords[c] is not None:
            df.loc[((df.cell_id == coords[c]) & (df.field == field) & (df.year == int(year))), column_name] = \
                column_data[c]

    df.to_csv(path, index=False)

    print("Done! The following file was updated: " + path)


def add_rotation_flip(x, y):
    # Flip horizontally
    x_h = np.flip(x[:, :, :, :, :], 3)
    # Flip vertically
    x_v = np.flip(x[:, :, :, :, :], 4)
    # Flip horizontally and vertically
    x_hv = np.flip(x_h[:, :, :, :, :], 4)

    # Concatenate
    x = np.concatenate((x, x_hv, x_v))
    y = np.concatenate((y, y, y))

    return x, y


def normalize(trainx):
    """Normalize and returns the calculated means and stds for each band"""
    trainxn = trainx.copy()
    means = np.zeros((trainx.shape[2], 1))
    stds = np.zeros((trainx.shape[2], 1))
    for n in range(trainx.shape[2]):
        means[n, ] = np.mean(trainxn[:, :, n, :, :])
        stds[n, ] = np.std(trainxn[:, :, n, :, :])
        trainxn[:, :, n, :, :] = (trainxn[:, :, n, :, :] - means[n, ]) / (stds[n, ])
    return trainxn, means, stds


def applynormalize(testx, means, stds):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    for n in range(testx.shape[2]):
        testxn[:, :, n, :, :] = (testxn[:, :, n, :, :] - means[n, ]) / (stds[n, ])
    return testxn


def minMaxScale(trainx):
    """Normalize and returns the calculated max and mins for each band"""
    trainxn = trainx.copy()
    maxs = np.zeros((trainx.shape[2], 1))
    mins = np.zeros((trainx.shape[2], 1))
    if trainx.ndim > 4:
        for n in range(trainx.shape[2]):
            maxs[n, ] = np.max(trainxn[:, :, n, :, :])
            mins[n, ] = np.min(trainxn[:, :, n, :, :])
            trainxn[:, :, n, :, :] = (trainxn[:, :, n, :, :] - mins[n, ]) / (maxs[n, ] - mins[n, ]) * 10
    else:
        maxs = np.max(trainxn)
        mins = np.min(trainxn)
        trainxn = (trainxn - mins) / (maxs - mins) * 10
    return trainxn, maxs, mins


def applyMinMaxScale(testx, maxs, mins):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if testxn.ndim > 4:
        for n in range(testx.shape[2]):
            testxn[:, :, n, :, :] = (testxn[:, :, n, :, :] - mins[n, ]) / (maxs[n, ] - mins[n, ]) * 10
    else:
        testxn = (testxn * (maxs - mins) / 10) + mins
    return testxn


def reversenormalize(testx, means, stds):
    """Reverse normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    for n in range(testx.shape[2]):
        testxn[:, :, n, :, :] = (testxn[:, :, n, :, :] * stds[n, ]) + means[n, ]
    return testxn


def reverseMinMaxScale(testx, maxs, mins):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if testxn.ndim > 4:
        for n in range(testx.shape[2]):
            testxn[:, :, n, :, :] = (testxn[:, :, n, :, :] * (maxs[n, ] - mins[n, ]) / 10) - mins[n, ]
    else:
        testxn = (testxn * (maxs - mins) / 10) + mins

    return testxn


def mse(imageA, imageB):
    """Calculate the 'Mean Squared Error' between the two images."""
    if imageA.ndim == 3:
        newdim = imageA.shape[0] * imageA.shape[1] * imageA.shape[2]
    elif imageA.ndim == 2:
        newdim = imageA.shape[0] * imageA.shape[1]
    else:
        sys.exit("Tensor has to have 2 o3 3 dimensions.")
    MSE = np.sum((np.reshape(imageA, newdim) - np.reshape(imageB, newdim)) ** 2) / float(newdim)
    return MSE


def rmsespatial(imageA, imageB):
    """Calculate the 'Mean Squared Error' between the two images pixel per pixel."""
    cells = imageA.shape[1] * imageA.shape[2]
    RMSE = np.zeros(cells)
    count = 0
    for i in range(imageA.shape[1]):
        for j in range(imageA.shape[2]):
            RMSE[count] = np.sqrt(np.sum((imageA[:, i, j] - imageB[:, i, j]) ** 2) / float(imageA.shape[0]))
            count += 1
    return RMSE


def MPIW_PICP(y_true, y_u=None, y_l=None, ypred=None, unc=None):
    """Calculate Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).
    @param y_true: Ground truth.
    @param y_u: Upper bound.
    @param y_l: Lower bound.
    @param ypred: Actual prediction. If y_u and y_l are not provided.
    @param unc: Standard error used to calculate y_u and y_l (e.g., y_u = ypred + unc)
    """
    # If hte actual predictions with uncertainties are given, calculate upper and lower bounds:
    if ypred is not None and unc is not None:
        y_u = ypred + unc
        y_l = ypred - unc

    # Calculate captured vector
    K_U = np.maximum(np.zeros(y_true.shape), np.sign(y_u - y_true))
    K_L = np.maximum(np.zeros(y_true.shape), np.sign(y_true - y_l))
    K = K_U * K_L

    MPIW = np.mean(y_u - y_l)
    MPIWcapt = np.sum((y_u - y_l) * K) / (np.sum(K) + 0.0001)
    PICP = np.mean(K)

    return MPIW, MPIWcapt, PICP, np.sum(K), len(K)
