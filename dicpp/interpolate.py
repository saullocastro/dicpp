r"""
Interpolate (:mod:`dicpp.interpolate`)
==================================================

.. currentmodule:: dicpp.interpolate

This module includes some interpolation utilities.

"""
import numpy as np
from numpy import sin, cos, tan
from scipy.spatial import cKDTree

from .logger import *


def inv_weighted(values, source, destiny, ncp=5, power_parameter=2):
    r"""Interpolates the values taken at one group of points into
    another using an inverse-weighted algorithm

    In the inverse-weighted algorithm a number of `n_{CP}` measured points
    of the input parameter ``source`` that are closest to a given node in
    the input parameter ``destiny`` are found and the ``values`` are
    interpolated as follows:

    .. math::
        {w_0}_{node} = \frac{\sum_{i}^{n_{CP}}{{w_0}_i\frac{1}{w_i}}}
                            {\sum_{i}^{n_{CP}}{\frac{1}{w_i}}}

    where `w_i` is the inverse weight of each measured point, calculated as:

    .. math::
        w_i = \left[(x_{node}-x_i)^2+(y_{node}-y_i)^2+(z_{node}-z_i)^2
              \right]^p

    with `p` being a power parameter that when increased will increase the
    relative influence of a closest point.


    Parameters
    ----------
    values: np.ndarray, shape (N)
        Values to be interpolated
    source : numpy.ndarray, shape (N, ndim)
        Source points corresponding to "values".
    destiny : numpy.ndarray, shape (M, ndim)
        The new coordinates where the values will be interpolated to.
    ncp : int, optional
        Number of closest points used in the inverse-weighted interpolation.
    power_parameter : float, optional
        Power of inverse weighted interpolation function.


    Returns
    -------
    dist, data_new : numpy.ndarray, numpy.ndarray
        Two 1-D arrays with the distances and interpolated values. The size of
        this array is ``destiny.shape[0]``.

    """
    if values.shape[0] != source.shape[0]:
        raise ValueError('Invalid input: values.shape[0] != source.shape[0]')
    if destiny.shape[1] != source.shape[1]:
        raise ValueError('Invalid input: source.shape[1] != destiny.shape[1]')

    tree = cKDTree(source)
    dist, indices = tree.query(destiny, k=ncp)

    # avoiding division by zero
    dist[dist == 0] = 1.e-12
    # fetching the data
    data = values[indices]
    # weight calculation
    total_weight = np.sum(1./(dist**power_parameter), axis=1)
    weight = 1./(dist**power_parameter)
    # computing the new data
    data_new = np.sum(data*weight, axis=1)/total_weight

    return dist, data_new


def interp_theta_z_imp(data, destiny, alphadeg, H_measured, H_model, R_bottom,
        stretch_H=False, z_offset_bot=None, rotatedeg=0., ncp=5,
        power_parameter=2, ignore_bot_h=None, ignore_top_h=None):
    r"""Interpolates a data set in the `\theta, z, imp` format

    This function uses the inverse-weighted algorithm (:func:`.inv_weighted`).

    Parameters
    ----------
    data : numpy.ndarray, shape (N, 3)
        The data or an array containing the imperfection file in the `(\theta,
        Z, imp)` format.
    destiny : numpy.ndarray, shape (M, 3)
        The destiny coordinates `(x, y, z)` where the values will be interpolated
        to.
    alphadeg : float
        The cone semi-vertex angle in degrees.
    H_measured : float
        The total height of the measured test specimen, including eventual
        resin rings at the edges.
    H_model : float
        The total height of the new model, including eventual resin rings at
        the edges.
    R_bottom : float
        The radius of the model taken at the bottom edge.
    stretch_H : bool, optional
        Tells if the height of the measured points, which is usually smaller
        than the height of the test specimen, should be stretched to fill
        the whole test specimen. If not, the points will be placed in the
        middle or using the offset given by ``z_offset_bot`` and the
        area not covered by the measured points will be interpolated
        using the closest available points (the imperfection
        pattern will look like there was an extrusion close to the edges).
    z_offset_bot : float, optional
        The offset that should be used from the bottom of the measured points
        to the bottom of the test specimen.
    rotatedeg : float, optional
        Rotation angle in degrees telling how much the imperfection pattern
        should be rotated about the `X_3` (or `Z`) axis.
    ncp : int, optional
        Number of closest points used in the inverse-weighted interpolation.
    power_parameter : float, optional
        Power of inverse weighted interpolation function.
    ignore_bot_h : None or float, optional
        Nodes close to the bottom edge are ignored according to this
        meridional distance.
    ignore_top_h : None or float, optional
        Nodes close to the top edge are ignored according to this meridional
        distance.

    Returns
    -------
    ans : numpy.ndarray
        An array with M elements containing the interpolated values.

    """
    assert data.shape[1] == 3, 'data must have shape (N, 3)'

    if stretch_H:
        H_points = data[:, 1].max() - data[:, 1].min()
        data[:, 1] /= H_points
    else:
        data[:, 1] /= H_measured

    if destiny.shape[1] != 3:
        raise ValueError('Mesh must have shape (M, 3)')

    data3D = np.zeros((data.shape[0], 4), dtype=np.float64)

    if rotatedeg:
        data[:, 0] += np.deg2rad(rotatedeg)

    z = data[:, 1]
    z *= H_model

    alpharad = np.deg2rad(alphadeg)
    tana = tan(alpharad)

    def r_local(z):
        return R_bottom - z*tana

    data3D[:, 0] = r_local(z)*cos(data[:, 0])
    data3D[:, 1] = r_local(z)*sin(data[:, 0])
    data3D[:, 2] = z
    data3D[:, 3] = data[:, 2]

    ans = inv_weighted(data3D, destiny, ncp=ncp, power_parameter=power_parameter)

    z_mesh = destiny[:, 2]
    if ignore_bot_h is not None:
        ans[(z_mesh - z_mesh.min()) <= ignore_bot_h] = 0.
    if ignore_top_h is not None:
        ans[(z_mesh.max() - z_mesh) <= ignore_top_h] = 0.

    return ans
