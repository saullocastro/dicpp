r"""
Fitting Data (:mod:`dicpostproc.fit_data`)
==================================================

.. currentmodule:: dicpostproc.fit_data

This module includes functions used to fit measured imperfection data.

"""
from random import sample
import os

import numpy as np
from numpy import sin, cos, pi, deg2rad
from scipy.sparse.linalg import aslinearoperator, LinearOperator
from scipy.optimize import least_squares, lsq_linear

from .logger import *
from .constants import FLOAT

def calc_Rx(a):
    """Rotation matrix around X for a 3D vector
    """
    return np.array([[1,      0,       0],
                     [0, cos(a), -sin(a)],
                     [0, sin(a),  cos(a)]])

def calc_Ry(b):
    """Rotation matrix around Y for a 3D vector
    """
    return np.array([[cos(b),  0, sin(b)],
                     [0,       1,      0],
                     [-sin(b), 0, cos(b)]])

def calc_Rz(c):
    """Rotation matrix around Z for a 3D vector
    """
    return np.array([[cos(c),  -sin(c), 0],
                     [sin(c),   cos(c), 0],
                     [     0,        0, 1]])

def calc_radius_ellipse(a, b, thetarad):
    t = thetarad
    return  a*b/np.sqrt((a*np.sin(t))**2 + (b*np.cos(t))**2)


def best_fit_cylinder(path, H, R_expected=10., save=True, errorRtol=1.e-9,
        maxNumIter=1000, sample_size=None, alpha0=0.5, beta0=0.5, x0=None,
        y0=None, z0=None, clip_box=None, loadtxtkwargs={}, ftol=1e-8,
        xtol=1e-8, gtol=1e-8):
    r"""Fit a best cylinder for a given set of measured data

    The coordinate transformation which must be performed in order to adjust
    the raw data to the finite element coordinate system is illustrated below:

    .. figure:: ../../../figures/modules/conecylDB/fit_data/coord_sys_trans.png
        :width: 400

    This transformation can be represented in matrix form as:

    .. math::

     {x, y, z}.T = [Ry][Rx]{x+x0, y+y0, z+z0}.T

    Note that **five** variables are unknowns:

    - the three components of the translation `\Delta x_0`, `\Delta y_0` and
      `\Delta z_0`
    - the rotation angles `\alpha` and `\beta`, respectively in this order

    The five unknowns are calculated iteratively in a non-linear least-squares
    problem (solved with ``scipy.optimize.least_squares``), where the measured data
    is transformed to the reference coordinate system and there compared with
    a reference cylinder in order to compute the residual error using:

    .. math::
        \begin{Bmatrix} x_{ref} \\ y_{ref} \\ z_{ref} \end{Bmatrix} =
        [T]
        \begin{Bmatrix} x_m \\ y_m \\ z_m \\ 1 \end{Bmatrix}
        \\
        Error = \sqrt{(\Delta r)^2 + (\Delta z)^2}

    where:

    - `x_m`, `y_m` and `z_m` are the data coordinates in the data coordinate
      system
    - `x_{ref}` `x_{ref}` are the data coordinates in the :ref:`reference
      coordinate system <figure_conecyl>`
    - `\Delta r` and `\Delta z` are defined as:

        .. math::
            \Delta r = R - \sqrt{x_{ref}^2 + y_{ref}^2}
            \\
            \Delta z = \begin{cases}
                            -z_{ref}, & \text{if } z_{ref} < 0 \\
                                   0, & \text{if } 0 <= z_{ref} <= H \\
                         z_{ref} - H, & \text{if } z_{ref} > H \\
                       \end{cases}

    Since the measured data may have an unknown radius `R`, the solution of
    these equations has to be performed iteratively with one additional
    external loop in order to update `R`.

    Parameters
    ----------
    path : str or np.ndarray
        The path of the file containing the data. Can be a full path using
        ``r"C:\Temp\inputfile.txt"``, for example.
        The input file must have 3 columns "`x` `y` `z`" expressed
        in Cartesian coordinates.

        This input can also be a ``np.ndarray`` object, with `x`, `y`, `z`
        in each corresponding column.
    H : float
        The nominal height of the cylinder.
    R_expected : float, optional
        The nominal radius of the cylinder, used as a first guess to find
        the best-fit radius (``R_best_fit``). Note that if not specified more
        iterations may be required.
    save : bool, optional
        Whether to save an ``"output_best_fit.txt"`` in the working directory.
    errorRtol : float, optional
        The error tolerance for the best-fit radius to stop the iterations.
    maxNumIter : int, optional
        The maximum number of iterations for the best-fit radius.
    sample_size : int, optional
        If the input file containing the measured data is too big it may
        be convenient to use only a sample of it in order to calculate the
        best fit.
    alpha0, beta0, x0, y0 ,z0: float, optional
        Initial guess for alpha, beta, x0, y0, z0.
    clip_box : None or sequence, optional
        Clip input points into [xmin, xmax, ymin, ymax, zmin, zmax]
    loadtxtkwargs : dict, optional
        Keyword arguments passed to ``np.loadtxt``
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. Default
        is 1e-8. The optimization process is stopped when  ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step. If None, the termination by this
        condition is disabled.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Default is 1e-8. The exact condition depends on the `method` used:

            * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``
            * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
              a trust-region radius and ``xs`` is the value of ``x``
              scaled according to `x_scale` parameter (see below).

        If None, the termination by this condition is disabled.
    gtol : float or None, optional
        Tolerance for termination by the norm of the gradient. Default is 1e-8.
        The exact condition depends on a `method` used:

            * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
              ``g_scaled`` is the value of the gradient scaled to account for
              the presence of the bounds [STIR]_.
            * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
              ``g_free`` is the gradient with respect to the variables which
              are not in the optimal state on the boundary.
            * For 'lm' : the maximum absolute value of the cosine of angles
              between columns of the Jacobian and the residual vector is less
              than `gtol`, or the residual vector is zero.

        If None, the termination by this condition is disabled.
    Returns
    -------
    out : dict
        A Python dictionary with the entries:

        ``out['R_best_fit']`` : float
            The best-fit radius of the input sample.
        ``out['x0']`` : float
            Translation in x
        ``out['y0']`` : float
            Translation in y
        ``out['z0']`` : float
            Translation in z
        ``out['alpharad']`` : np.ndarray
            `\alpha` angle to rotate input data.
        ``out['betarad']`` : np.ndarray
            `\beta` angle to rotate input data.
        ``out['input_pts']`` : np.ndarray
            The input points in a `3 \times N` 2-D array.
        ``out['output_pts']`` : np.ndarray
            The transformed points in a `3 \times N` 2-D array.

    Examples
    --------

    1) General usage

    For a given cylinder with expected radius and height of ``R_expected`` and
    ``H``::

        from dicpostproc.fit_data import best_fit_cylinder

        out = best_fit_cylinder(path, H=H, R_expected=R_expected)
        R_best_fit = out['R_best_fit']

    2) Using the transformation data

    For a given input data with `x, y, z` positions in each line::

        x, y, z = np.loadtxt('input_file.txt', unpack=True)

    the transformation can be obtained with::

        Rx = calc_Rx(alpharad)
        Ry = calc_Ry(betarad)
        xnew, ynew, znew = (Ry @ Rx).dot(np.vstack((x + x0, y + y0, z + z0)))

    and the inverse transformation::

        x, y, z = (Rx.T @ Ry.T).dot(np.vstack((xnew, ynew, znew))) - np.array([x0, y0, y0])



    """
    if isinstance(path, np.ndarray):
        input_pts = path.T
    else:
        input_pts = np.loadtxt(path, unpack=True, **loadtxtkwargs)

    if input_pts.shape[0] != 3:
        raise ValueError('Input does not have the format: "x, y, z"')

    if sample_size:
        num = input_pts.shape[1]
        if sample_size < num:
            input_pts = input_pts[:, sample(range(num), int(sample_size))]

    if clip_box is not None:
        assert len(clip_box) == 6, 'Clip box must be [xmin, xmax, ymin, ymax, zmin, zmax]'
        x, y, z = input_pts
        valid = ((clip_box[0] <= x) &
                 (x <= clip_box[1]) &
                 (clip_box[2] <= y) &
                 (y <= clip_box[3]) &
                 (clip_box[4] <= z) &
                 (z <= clip_box[5]))
        input_pts = input_pts[:, valid]


    i = 0
    R = R_expected
    while i <= maxNumIter:
        i += 1

        def calc_dist(p, input_pts):
            Rx = calc_Rx(p[0])
            Ry = calc_Ry(p[1])
            x0, y0, z0 = p[2:]
            xn, yn, zn = Ry.dot(Rx.dot(input_pts + np.array([x0, y0, z0])[:, None]))
            dz = np.zeros_like(zn)
            factor = 1
            # point below the bottom edge
            mask = zn < 0
            dz[mask] = -zn[mask]*factor

            # point inside the cylinder
            pass
            #dz[(zn >= 0) & (zn <= H)] *= 0

            # point above the top edge
            mask = zn > H
            dz[mask] = (zn[mask] - H)*factor

            dr = R - np.sqrt(xn**2 + yn**2)
            dist = np.sqrt(dr**2 + dz**2)
            return dist

        # initial guess for the optimization variables
        # the variables are alpha, beta, x0, y0, z0
        x, y, z = input_pts
        if x0 is None:
            x0 = 2*x.mean()
        if y0 is None:
            y0 = 2*y.mean()
        if z0 is None:
            z0 = 2*z.mean()
        p = [alpha0, beta0, x0, y0, z0]

        # performing the least_squares analysis
        bounds = ((-np.pi, -np.pi, -np.inf, -np.inf, -np.inf),
                  (np.pi, np.pi, np.inf, np.inf, np.inf))
        res = least_squares(fun=calc_dist, x0=p, bounds=bounds, args=(input_pts,),
                             max_nfev=1000000, xtol=xtol, ftol=ftol, gtol=gtol)
        popt = res.x
        Rx = calc_Rx(popt[0])
        Ry = calc_Ry(popt[1])
        x0, y0, z0 = popt[2:]
        output_pts = Ry.dot(Rx.dot(input_pts + np.array([x0, y0, z0])[:, None]))
        x, y, z = output_pts
        mask = (z>=0) & (z<=H)
        R_best_fit = np.sqrt(x[mask]**2 + y[mask]**2).mean()
        errorR = abs(R_best_fit - R)/R_best_fit

        log('Iteration: {0}, R_best_fit: {1}, errorR: {2}'.format(
            i, R_best_fit, errorR), level=1)

        if errorR < errorRtol:
            break
        else:
            R = R_best_fit
    else:
        warn('The maximum number of iterations was achieved!')

    alpha, beta, x0, y0, z0 = popt
    log('Translation:')
    log('x0, y0, z0: {0}, {1}, {2}'.format(x0, y0, z0))
    log('')
    log('Rotation angles:')
    log('alpha: {0} rad; beta: {1} rad'.format(alpha, beta))
    log('')
    log('Best fit radius: {0}'.format(R_best_fit))
    log('    errorR: {0}, numiter: {1}'.format(errorR, i))
    log('')

    if save:
        np.savetxt('output_best_fit.txt', np.vstack((x, y, z)).T)

    return dict(R_best_fit=R_best_fit,
                input_pts=input_pts,
                output_pts=output_pts,
                alpharad=alpha,
                betarad=beta,
                x0=x0, y0=y0, z0=z0)


def best_fit_elliptic_cylinder(path, H, a_expected=10., b_expected=10.,
        save=True, alpha0=0.5, beta0=0.5, gamma0=0., x0=None,
        y0=None, z0=None, clip_box=None, loadtxtkwargs={}, ftol=1e-8,
        xtol=1e-8, gtol=1e-8, a_min=-1e6, a_max=1e6, b_min=-1e6, b_max=1e6):
    r"""Fit a best cylinder for a given set of measured data

    The coordinate transformation which must be performed in order to adjust
    the raw data to the finite element coordinate system is illustrated below:

    .. figure:: ../../../figures/modules/conecylDB/fit_data/coord_sys_trans.png
        :width: 400

    This transformation can be represented in matrix form as:

    .. math::

     {x, y, z}.T = [Rz][Ry][Rx]{x+x0, y+y0, z+z0}.T

    Note that **five** variables are unknowns:

    - the three components of the translation `\Delta x_0`, `\Delta y_0` and `\Delta z_0`
    - the rotation angles `\alpha`, `\beta` and `\gamma`; respectively in this order

    The five unknowns are calculated iteratively in a non-linear least-squares
    problem (solved with ``scipy.optimize.least_squares``), where the measured data
    is transformed to the reference coordinate system and there compared with
    a reference cylinder in order to compute the residual error using:

    .. math::
        \begin{Bmatrix} x_{ref} \\ y_{ref} \\ z_{ref} \end{Bmatrix} =
        [T]
        \begin{Bmatrix} x_m \\ y_m \\ z_m \\ 1 \end{Bmatrix}
        \\
        Error = \sqrt{(\Delta r)^2 + (\Delta z)^2}

    where:

    - `x_m`, `y_m` and `z_m` are the data coordinates in the data coordinate
      system
    - `x_{ref}` `x_{ref}` are the data coordinates in the :ref:`reference
      coordinate system <figure_conecyl>`
    - `\Delta r` and `\Delta z` are defined as:

        .. math::
            \Delta r = R - \sqrt{x_{ref}^2 + y_{ref}^2}
            \\
            \Delta z = \begin{cases}
                            -z_{ref}, & \text{if } z_{ref} < 0 \\
                                   0, & \text{if } 0 <= z_{ref} <= H \\
                         z_{ref} - H, & \text{if } z_{ref} > H \\
                       \end{cases}

    Since the measured data may have an unknown radius `R`, the solution of
    these equations has to be performed iteratively with one additional
    external loop in order to update `R`.

    Parameters
    ----------
    path : str or np.ndarray
        The path of the file containing the data. Can be a full path using
        ``r"C:\Temp\inputfile.txt"``, for example.
        The input file must have 3 columns "`x` `y` `z`" expressed
        in Cartesian coordinates.

        This input can also be a ``np.ndarray`` object, with `x`, `y`, `z`
        in each corresponding column.
    H : float
        The nominal height of the cylinder.
    a_expected, a_min, a_max : float, optional
        The major radius of the elliptic cylinder, used as a first guess to find
        the best-fit major radius (``a_best_fit``). Note that if not specified more
        iterations may be required.
    b_expected, b_min, b_max : float, optional
        The minor radius of the elliptic cylinder, used as a first guess to find
        the best-fit minor radius (``b_best_fit``). Note that if not specified more
        iterations may be required.
    save : bool, optional
        Whether to save an ``"output_best_fit.txt"`` in the working directory.
    alpha0, beta0, gamma0, x0, y0 ,z0: float, optional
        Initial guess for alpha, beta, gamma, x0, y0, z0.
    clip_box : None or sequence, optional
        Clip input points into [xmin, xmax, ymin, ymax, zmin, zmax]
    loadtxtkwargs : dict, optional
        Keyword arguments passed to ``np.loadtxt``
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. Default
        is 1e-8. The optimization process is stopped when  ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step. If None, the termination by this
        condition is disabled.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Default is 1e-8. The exact condition depends on the `method` used:

            * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``
            * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
              a trust-region radius and ``xs`` is the value of ``x``
              scaled according to `x_scale` parameter (see below).

        If None, the termination by this condition is disabled.
    gtol : float or None, optional
        Tolerance for termination by the norm of the gradient. Default is 1e-8.
        The exact condition depends on a `method` used:

            * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
              ``g_scaled`` is the value of the gradient scaled to account for
              the presence of the bounds [STIR]_.
            * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
              ``g_free`` is the gradient with respect to the variables which
              are not in the optimal state on the boundary.
            * For 'lm' : the maximum absolute value of the cosine of angles
              between columns of the Jacobian and the residual vector is less
              than `gtol`, or the residual vector is zero.

        If None, the termination by this condition is disabled.

    Returns
    -------
    out : dict
        A Python dictionary with the entries:

        ``out['a_best_fit']``, ``out['b_best_fit']``: float
            The best-fit radii of the input sample.
        ``out['x0']`` : float
            Translation in x
        ``out['y0']`` : float
            Translation in y
        ``out['z0']`` : float
            Translation in z
        ``out['alpharad']`` : np.ndarray
            `\alpha` angle to rotate input data.
        ``out['betarad']`` : np.ndarray
            `\beta` angle to rotate input data.
        ``out['gammarad']`` : np.ndarray
            `\gamma` angle to rotate input data.
        ``out['input_pts']`` : np.ndarray
            The input points in a `3 \times N` 2-D array.
        ``out['output_pts']`` : np.ndarray
            The transformed points in a `3 \times N` 2-D array.

    Examples
    --------

    1) General usage

    For a given elliptic cylinder with expected radii and height of ``a_expected``,
    ``b_expected`` and ``H``::

        from dicpostproc.fit_data import best_fit_elliptic_cylinder

        out = best_fit_elliptic_cylinder(path, H=H, a_expected=a_expected,
        b_expected=b_expected)
        a_best_fit = out['a_best_fit']
        b_best_fit = out['b_best_fit']

    2) Using the transformation data

    For a given input data with `x, y, z` positions in each line::

        x, y, z = np.loadtxt('input_file.txt', unpack=True)

    the transformation can be obtained with::

        Rx = calc_Rx(alpharad)
        Ry = calc_Ry(betarad)
        Rz = calc_Rz(gammarad)
        xnew, ynew, znew = (Rz @ Ry @ Rx).dot(np.vstack((x + x0, y + y0, z + z0)))

    and the inverse transformation::

        x, y, z = (Rx.T @ Ry.T @ Rz.T).dot(np.vstack((xnew, ynew, znew))) - np.array([x0, y0, y0])



    """
    if isinstance(path, np.ndarray):
        input_pts = path.T
    else:
        input_pts = np.loadtxt(path, unpack=True, **loadtxtkwargs)

    if input_pts.shape[0] != 3:
        raise ValueError('Input does not have the format: "x, y, z"')

    if clip_box is not None:
        assert len(clip_box) == 6, 'Clip box must be [xmin, xmax, ymin, ymax, zmin, zmax]'
        x, y, z = input_pts
        valid = ((clip_box[0] <= x) &
                 (x <= clip_box[1]) &
                 (clip_box[2] <= y) &
                 (y <= clip_box[3]) &
                 (clip_box[4] <= z) &
                 (z <= clip_box[5]))
        input_pts = input_pts[:, valid]


    i = 0
    a = a_expected
    b = b_expected

    def calc_dist_cylinder(p, input_pts):
        Rx = calc_Rx(p[0])
        Ry = calc_Ry(p[1])
        x0, y0, z0 = p[2:]
        xn, yn, zn = Ry @ Rx @ (input_pts + np.array([x0, y0, z0])[:, None])
        t = np.arctan2(yn, xn)
        dr = np.sqrt(xn**2 + yn**2) - a_expected
        dist = np.sqrt(dr*dr)
        return dist

    def calc_dist_ellipse(p, interm_pts):
        Rz = calc_Rz(p[0])
        a, b = p[1:]
        xn, yn, zn = Rz @ interm_pts
        t = np.arctan2(yn, xn)
        rellipse = calc_radius_ellipse(a, b, t)
        dr = np.sqrt(xn**2 + yn**2) - rellipse
        dist = np.sqrt(dr*dr)
        return dist

    def calc_dist_dz(z_offset, interm_pts):
        xn, yn, zn = interm_pts
        zn = zn + z_offset
        dz = np.zeros_like(zn)
        # point below the bottom edge
        mask = zn < 0
        dz[mask] = -zn[mask]
        # point inside the cylinder
        pass
        #dz[(zn >= 0) & (zn <= H)] *= 0
        # point above the top edge
        mask = zn > H
        dz[mask] = (zn[mask] - H)
        dist = np.sqrt(dz*dz)
        return dist

    # initial guess for the optimization variables
    # the variables are alpha, beta, x0, y0, z0
    x, y, z = input_pts
    if x0 is None:
        x0 = 2*x.mean()
    if y0 is None:
        y0 = 2*y.mean()
    if z0 is None:
        z0 = 2*z.mean()

    # performing the least_squares analysis
    p = [alpha0, beta0, x0, y0, z0]
    bounds = ((-np.pi, -np.pi, -np.inf, -np.inf, -np.inf),
              (np.pi, np.pi, np.inf, np.inf, np.inf))
    res = least_squares(fun=calc_dist_cylinder, x0=p, bounds=bounds,
            args=(input_pts,), max_nfev=1000000, xtol=xtol, ftol=ftol,
            gtol=gtol, jac='3-point')
    print('least_squares status', res.message)
    popt = res.x
    alpha = popt[0]
    beta = popt[1]
    Rx = calc_Rx(alpha)
    Ry = calc_Ry(beta)
    x0, y0, z0 = popt[2:]
    interm_pts1 = (Ry @ Rx @ (input_pts + np.array([x0, y0, z0])[:, None]))

    p = [gamma0, a_expected, b_expected]
    bounds = ((-np.pi/2, a_min, b_min),
              (+np.pi/2, a_max, b_max))
    res = least_squares(fun=calc_dist_ellipse, x0=p, bounds=bounds,
            args=(interm_pts1, ), max_nfev=1000000, xtol=xtol, ftol=ftol,
            gtol=gtol, jac='3-point')
    print('least_squares status', res.message)
    popt = res.x
    gamma = popt[0]
    Rz = calc_Rz(gamma)
    a_best_fit, b_best_fit = popt[1:]
    interm_pts2 = Rz @ interm_pts1

    z_offset = -1.
    bounds = ([-np.inf], [+np.inf])
    res = least_squares(fun=calc_dist_dz, x0=z_offset, bounds=bounds,
            args=(interm_pts2, ), max_nfev=1000000, xtol=xtol, ftol=ftol,
            gtol=gtol, jac='3-point')
    z_offset = res.x
    interm_pts2[2] += z_offset

    output_pts = interm_pts2
    x, y, z = output_pts

    log('Translation:')
    log('x0, y0, z0: {0}, {1}, {2}'.format(x0, y0, z0))
    log('')
    log('Rotation angles:')
    log('alpha: {0} rad; beta: {1} rad; gamma: {2} rad'.format(alpha, beta, gamma))
    log('')
    log('Longitudinal offset:')
    log('z_offset: {0}'.format(z_offset))
    log('')
    log('Best fit radii: a={0}, b={1}'.format(a_best_fit, b_best_fit))
    log('')

    if save:
        np.savetxt('output_best_fit.txt', np.vstack((x, y, z)).T)

    return dict(a_best_fit=a_best_fit,
                b_best_fit=b_best_fit,
                input_pts=input_pts,
                output_pts=output_pts,
                alpharad=alpha,
                betarad=beta,
                gammarad=gamma,
                x0=x0, y0=y0, z0=z0, z_offset=z_offset)


def best_fit_cone(path, H, alphadeg, R_expected=10., save=True,
        errorRtol=1.e-9, maxNumIter=1000, sample_size=None):
    r"""Fit a best cone for a given set of measured data

    .. note:: NOT IMPLEMENTED YET

    """
    raise NotImplementedError('Function not implemented yet!')


def calc_c0(path, m0=50, n0=50, funcnum=2, fem_meridian_bot2top=True,
        rotatedeg=None, filter_m0=None, filter_n0=None, sample_size=None,
        loadtxtkwargs={}):
    r"""Find the coefficients that best fit the `w_0` imperfection

    The measured data will be fit using one of the following functions,
    selected using the ``funcnum`` parameter:

    1) Half-Sine Function

    .. math::
        w_0 = \sum_{i=1}^{m_0}{ \sum_{j=0}^{n_0}{
                 {c_0}_{ij}^a sin{b_z} sin{b_\theta}
                +{c_0}_{ij}^b sin{b_z} cos{b_\theta} }}

    2) Half-Cosine Function (default)

    .. math::
        w_0 = \sum_{i=0}^{m_0}{ \sum_{j=0}^{n_0}{
                {c_0}_{ij}^a cos{b_z} sin{b_\theta}
                +{c_0}_{ij}^b cos{b_z} cos{b_\theta} }}

    3) Complete Fourier Series

    .. math::
        w_0 = \sum_{i=0}^{m_0}{ \sum_{j=0}^{n_0}{
                 {c_0}_{ij}^a sin{b_z} sin{b_\theta}
                +{c_0}_{ij}^b sin{b_z} cos{b_\theta}
                +{c_0}_{ij}^c cos{b_z} sin{b_\theta}
                +{c_0}_{ij}^d cos{b_z} cos{b_\theta} }}

    where:

    .. math::
        b_z = i \pi \frac z H_{points}

        b_\theta = j \theta

    where `H_{points}` represents the difference between the maximum and
    the minimum `z` values in the imperfection file.

    The approximation can be written in matrix form as:

    .. math::
        w_0 = [g] \{c_0\}

    where `[g]` carries the base functions and `{c_0}` the respective
    amplitudes. The solution consists on finding the best `{c_0}` that
    minimizes the least-square error between the measured imperfection pattern
    and the `w_0` function.

    Parameters
    ----------
    path : str or np.ndarray
        The path of the file containing the data. Can be a full path using
        ``r"C:\Temp\inputfile.txt"``, for example.
        The input file must have 3 columns "`\theta` `z` `imp`" expressed
        in Cartesian coordinates.

        This input can also be a ``np.ndarray`` object, with
        `\theta`, `z`, `imp` in each corresponding column.
    m0 : int
        Number of terms along the meridian (`z`).
    n0 : int
        Number of terms along the circumference (`\theta`).
    funcnum : int, optional
        As explained above, selects the base functions used for
        the approximation.
    fem_meridian_bot2top : bool, optional
        A boolean indicating if the finite element has the `x` axis starting
        at the bottom or at the top.
    rotatedeg : float or None, optional
        Rotation angle in degrees telling how much the imperfection pattern
        should be rotated about the `X_3` (or `Z`) axis.
    filter_m0 : list, optional
        The values of ``m0`` that should be filtered (see :func:`.filter_c0`).
    filter_n0 : list, optional
        The values of ``n0`` that should be filtered (see :func:`.filter_c0`).
    sample_size : int or None, optional
        An in  specifying how many points of the imperfection file should
        be used. If ``None`` is used all points file will be used in the
        computations.
    loadtxtkwargs : dict, optional
        Keyword arguments passed to ``np.loadtxt``

    Returns
    -------
    out : np.ndarray
        A 1-D array with the best-fit coefficients.

    Notes
    -----
    If a similar imperfection pattern is expected along the meridian and along
    the circumference, the analyst can use an optimized relation between
    ``m0`` and ``n0`` in order to achieve a higher accuracy for a given
    computational cost, as proposed by Castro et al. (2014):

    .. math::
        n_0 = m_0 \frac{\pi(R_{bot}+R_{top})}{2H}

    """
    from scipy.linalg import lstsq

    if isinstance(path, np.ndarray):
        input_pts = path
        path = 'unmamed.txt'
    else:
        input_pts = np.loadtxt(path, **loadtxtkwargs)

    if input_pts.shape[1] != 3:
        raise ValueError('Input does not have the format: "theta, z, imp"')
    if (input_pts[:,0].min() < -2*np.pi or input_pts[:,0].max() > 2*np.pi):
        raise ValueError(
                'In the input: "theta, z, imp"; "theta" must be in radians!')

    log('Finding c0 coefficients for {0}'.format(str(os.path.basename(path))))
    log('using funcnum {0}'.format(funcnum), level=1)

    if sample_size:
        num = input_pts.shape[0]
        if sample_size < num:
            input_pts = input_pts[sample(range(num), int(sample_size))]

    if funcnum==1:
        size = 2
    elif funcnum==2:
        size = 2
    elif funcnum==3:
        size = 4
    else:
        raise ValueError('Valid values for "funcnum" are 1, 2 or 3')

    ts = input_pts[:, 0].copy()
    if rotatedeg is not None:
        ts += deg2rad(rotatedeg)
    zs = input_pts[:, 1]
    w0pts = input_pts[:, 2]
    #NOTE using `H_measured` did not allow a good fitting result
    #zs /= H_measured
    zs = (zs - zs.min())/(zs.max() - zs.min())
    if not fem_meridian_bot2top:
        #TODO
        zs *= -1
        zs += 1

    a = fa(m0, n0, zs, ts, funcnum)
    A = aslinearoperator(a)
    log('Base functions calculated', level=1)
    #TODO calculate rmatvec
    #def matvec(x):
        #return fw0(m0, n0, x, xs_norm=zs, ts=ts, funcnum=funcnum)
    #def rmatvec(x):
        #if x.shape == size*m0*n0:
            #return w0pts
        #else:
            #return np.ones(size*m0*n0)
    #A = LinearOperator((ts.shape[0], size*m0*n0), matvec=matvec,
            #rmatvec=rmatvec)
    res = lsq_linear(A, w0pts)
    c0 = res.x
    residues = res.fun
    log('Finished scipy.optimize.lsq_linear', level=1)

    if filter_m0 is not None or filter_n0 is not None:
        c0 = filter_c0(m0, n0, c0, filter_m0, filter_n0, funcnum=funcnum)

    return c0, residues


def filter_c0(m0, n0, c0, filter_m0, filter_n0, funcnum=2):
    r"""Apply filter to the imperfection coefficients `\{c_0\}`

    A filter consists on removing some frequencies that are known to be
    related to rigid body modes or spurious measurement noise. The frequencies
    to be removed should be passed through inputs ``filter_m0`` and
    ``filter_n0``.

    Parameters
    ----------
    m0 : int
        The number of terms along the meridian.
    n0 : int
        The number of terms along the circumference.
    c0 : np.ndarray
        The coefficients of the imperfection pattern.
    filter_m0 : list
        The values of ``m0`` that should be filtered.
    filter_n0 : list
        The values of ``n0`` that should be filtered.
    funcnum : int, optional
        The function used for the approximation (see function :func:`.calc_c0`)

    Returns
    -------
    c0_filtered : np.ndarray
        The filtered coefficients of the imperfection pattern.

    """
    log('Applying filter...')
    log('using c0.shape={0}, funcnum={1}'.format(c0.shape, funcnum), level=1)
    fm0 = filter_m0
    fn0 = filter_n0
    log('using filter_m0={0}'.format(fm0))
    log('using filter_n0={0}'.format(fn0))
    if funcnum==1:
        if 0 in fm0:
            raise ValueError('For funcnum==1 m0 starts at 1!')
        pos = ([2*(m0*j + (i-1)) + 0 for j in range(n0) for i in fm0] +
               [2*(m0*j + (i-1)) + 1 for j in range(n0) for i in fm0])
        pos += ([2*(m0*j + (i-1)) + 0 for j in fn0 for i in range(1, m0+1)] +
                [2*(m0*j + (i-1)) + 1 for j in fn0 for i in range(1, m0+1)])
    elif funcnum==2:
        pos = ([2*(m0*j + i) + 0 for j in range(n0) for i in fm0] +
               [2*(m0*j + i) + 1 for j in range(n0) for i in fm0])
        pos += ([2*(m0*j + i) + 0 for j in fn0 for i in range(m0)] +
                [2*(m0*j + i) + 1 for j in fn0 for i in range(m0)])
    elif funcnum==3:
        pos = ([4*(m0*j + i) + 0 for j in range(n0) for i in fm0] +
               [4*(m0*j + i) + 1 for j in range(n0) for i in fm0] +
               [4*(m0*j + i) + 2 for j in range(n0) for i in fm0] +
               [4*(m0*j + i) + 3 for j in range(n0) for i in fm0])
        pos += ([4*(m0*j + i) + 0 for j in fn0 for i in range(m0)] +
                [4*(m0*j + i) + 1 for j in fn0 for i in range(m0)] +
                [4*(m0*j + i) + 2 for j in fn0 for i in range(m0)] +
                [4*(m0*j + i) + 3 for j in fn0 for i in range(m0)])
    c0_filtered = c0.copy()
    c0_filtered[pos] = 0
    log('Filter applied!')
    return c0_filtered


def fa(m0, n0, zs_norm, thetas, funcnum=2):
    """Calculates the matrix with the base functions for `w_0`

    The calculated matrix is directly used to calculate the `w_0` displacement
    field, when the corresponding coefficients `c_0` are known, through::

        a = fa(m0, n0, zs_norm, thetas, funcnum)
        w0 = a.dot(c0)

    Parameters
    ----------
    m0 : int
        The number of terms along the meridian.
    n0 : int
        The number of terms along the circumference.
    zs_norm : np.ndarray
        The normalized `z` coordinates (from 0. to 1.) used to compute
        the base functions.
    thetas : np.ndarray
        The angles in radians representing the circumferential positions.
    funcnum : int, optional
        The function used for the approximation (see function :func:`.calc_c0`)

    """
    try:
        from . import fit_data_core
        return fit_data_core.fa(m0, n0, zs_norm, thetas, funcnum)
    except:
        warn('fit_data_core.pyx could not be imported, executing in Python/NumPy'
                + '\n\t\tThis mode is slower and needs more memory than the'
                + '\n\t\tPython/NumPy/Cython mode',
             level=1)
        zs = zs_norm.ravel()
        ts = thetas.ravel()
        n = zs.shape[0]
        zsmin = zs.min()
        zsmax = zs.max()
        if zsmin < 0 or zsmax > 1:
            log('zs.min()={0}'.format(zsmin))
            log('zs.max()={0}'.format(zsmax))
            raise ValueError('The zs array must be normalized!')
        if funcnum==1:
            a = np.array([[sin(i*pi*zs)*sin(j*ts), sin(i*pi*zs)*cos(j*ts)]
                           for j in range(n0) for i in range(1, m0+1)])
            a = a.swapaxes(0,2).swapaxes(1,2).reshape(n,-1)
        elif funcnum==2:
            a = np.array([[cos(i*pi*zs)*sin(j*ts), cos(i*pi*zs)*cos(j*ts)]
                           for j in range(n0) for i in range(m0)])
            a = a.swapaxes(0,2).swapaxes(1,2).reshape(n,-1)
        elif funcnum==3:
            a = np.array([[sin(i*pi*zs)*sin(j*ts), sin(i*pi*zs)*cos(j*ts),
                           cos(i*pi*zs)*sin(j*ts), cos(i*pi*zs)*cos(j*ts)]
                           for j in range(n0) for i in range(m0)])
            a = a.swapaxes(0,2).swapaxes(1,2).reshape(n,-1)
    return a


def fw0(m0, n0, c0, xs_norm, ts, funcnum=2):
    r"""Calculates the imperfection field `w_0` for a given input

    Parameters
    ----------
    m0 : int
        The number of terms along the meridian.
    n0 : int
        The number of terms along the circumference.
    c0 : np.ndarray
        The coefficients of the imperfection pattern.
    xs_norm : np.ndarray
        The meridian coordinate (`x`) normalized to be between ``0.`` and
        ``1.``.
    ts : np.ndarray
        The angles in radians representing the circumferential coordinate
        (`\theta`).
    funcnum : int, optional
        The function used for the approximation (see function :func:`.calc_c0`)

    Returns
    -------
    w0s : np.ndarray
        An array with the same shape of ``xs_norm`` containing the calculated
        imperfections.

    Notes
    -----
    The inputs ``xs_norm`` and ``ts`` must be of the same size.

    The inputs must satisfy ``c0.shape[0] == size*m0*n0``, where:

    - ``size=2`` if ``funcnum==1 or funcnum==2``
    - ``size=4`` if ``funcnum==3``

    """
    if xs_norm.shape != ts.shape:
        raise ValueError('xs_norm and ts must have the same shape')
    if funcnum==1:
        size = 2
    elif funcnum==2:
        size = 2
    elif funcnum==3:
        size = 4
    if c0.shape[0] != size*m0*n0:
        raise ValueError('Invalid c0 (shape %s) for the given m0 and n0 (shape %s)!'
                % (str(c0.shape), str(size*m0*n0)))
    try:
        from . import fit_data_core
        w0s = fit_data_core.fw0(m0, n0, c0, xs_norm.ravel(), ts.ravel(), funcnum)
    except:
        a = fa(m0, n0, xs_norm.ravel(), ts.ravel(), funcnum)
        w0s = a.dot(c0)
    return w0s.reshape(xs_norm.shape)

