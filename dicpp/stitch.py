r"""
Fitting Data (:mod:`dicpp.stitch`)
==================================================

.. currentmodule:: dicpp.stitch

Functions to stitch best-fit imperfection data.

"""

import sys
import pickle

import numpy as np
from numpy import deg2rad
from scipy.optimize import least_squares

from dicpp.fit_data import calc_Rx, calc_Ry, calc_Rz, calc_radius_ellipse
from dicpp.interpolate import inv_weighted


def stitch(bf1, bf2,
        pos_deg1, pos_deg2, height_ref,
        probe_deg, probe_R, probe_zarray,
        init_deg_min=-10, init_deg_max=+10, init_deg_step=11,
        init_z_min=-20, init_z_max=+20, init_z_step=11,
        opt_var_z_min=-10., opt_var_z_max=+10., opt_var_deg_min=-10., opt_var_deg_max=+10.,
        ls_kwargs=dict(ftol=None, xtol=1e-4, gtol=None, method='trf',
            max_nfev=1000000, jac='3-point', diff_step=0.2)):
    r"""
    Calculate `\Delta \theta_2` and `\Delta z_2` that stitches the second best fit
    results to the first.

    Parameters
    ----------
    bf1, bf2 : dict
        Results returned by the best fit functions. If the results are for a
        best-fit elliptic cylinder, the imperfections are remapped to a
        cylinder of radius``R=out['a_best_fit']``.
    pos_deg1, pos_deg_2 : float
        Nominal circumferential position in degrees of the adjacent best fit
        results.
    height_ref : float
        Reference height.
    probe_deg : float
        Position in degrees of the probing line.
    probe_R : float
        Radial position of the probing line.
    probe_zarray: array-like
        Array with the z positions of the probing line.
    init_deg_min, init_deg_max, init_z_min, init_z_max : float, optional
        Interval to search for the best initial point for the stitching
        optimization.
    init_deg_step, init_z_step : int, optional
        Number of points in the interval to search for the best initial point
        for the stitching optimization.
    opt_var_z_min, opt_var_z_max, opt_var_deg_min, opt_var_deg_max : float, optional
        Variation from initial point to be used in the stitching optimization.
    ls_kwargs : dict, optional
        Keyword arguments passed to ``scipy.optimize.least_squares``.
    diff_step_deg : float, optional
        Step size used in the finite difference to compute gradients.

    Returns
    -------
    out : dict
        A Python dictionary with the entries:

        ``out['delta_deg2']`` : float
            The `\Delta \theta_2` offset that stitches the second best-fit
            result to the first.
        ``out['delta_z2']`` : float
            The `\Delta z_2` offset that stitches the second best-fit result to
            the first.
        ``out['dr1']`` : array-like
            Radial imperfection of the first best-fit result at the probing
            line.
        ``out['dr2']`` : array-like
            Radial imperfection of the second best-fit result at the probing
            line.

    """
    def get_xyz_imp(p, bf, pos_deg):
        delta_deg = p[0]
        delta_z = p[1]
        delta_rad = deg2rad(delta_deg)
        Rx = calc_Rx(bf['alpharad'])
        Ry = calc_Ry(bf['betarad'])
        Rz = calc_Rz(bf['gammarad'])
        x0 = bf['x0']
        y0 = bf['y0']
        z0 = bf['z0']
        z1 = bf['z1']
        xyz = bf['input_pts'].T
        xyz[:, 0] += x0
        xyz[:, 1] += y0
        xyz[:, 2] += z0
        xyz = ((Rz @ Ry @ Rx ) @ xyz.T).T
        xyz[:, 2] += z1

        xyz[:, 2] -= xyz[:, 2].mean()
        xyz[:, 2] += height_ref/2

        theta = np.arctan2(xyz[:, 1], xyz[:, 0])
        if 'a_best_fit' in bf:
            # remaping elliptic cylinder onto a cylinder or R=a
            a = bf['a_best_fit']
            b = bf['b_best_fit']
            R = a
            radius_ellipse = calc_radius_ellipse(a, b, theta)
            deltar = (xyz[:, 0]**2 + xyz[:, 1]**2)**0.5 - radius_ellipse
            rcnew = R + deltar
            xyz[:, 0] = rcnew*np.cos(theta)
            xyz[:, 1] = rcnew*np.sin(theta)
        else:
            deltar = (xyz[:, 0]**2 + xyz[:, 1]**2)**0.5 - bf['R_best_fit']
        #NOTE rotating face to nominal circumferential position
        Rz = calc_Rz(deg2rad(pos_deg))
        xyz = (Rz @ xyz.T).T
        #NOTE applying \Delta z
        xyz[:, 2] += delta_z
        #NOTE applying \Delta \Theta
        Rz = calc_Rz(delta_rad)
        xyz = (Rz @ xyz.T).T
        return xyz, deltar

    def dr_at_probing_line(xyz, dr):
        xyz_mesh = np.zeros((probe_zarray.shape[0], 3))
        xyz_mesh[:, 0] = probe_R*np.cos(deg2rad(probe_deg))
        xyz_mesh[:, 1] = probe_R*np.sin(deg2rad(probe_deg))
        xyz_mesh[:, 2] = probe_zarray
        dist, out = inv_weighted(dr, xyz, xyz_mesh, ncp=10, power_parameter=1.7)
        return out

    xyz1, deltar1 = get_xyz_imp([0, 0], bf1, pos_deg=pos_deg_1)
    dr1 = dr_at_probing_line(xyz1, deltar1)

    def fun(p):
        xyz2, deltar2 = get_xyz_imp(p, bf2, pos_deg=pos_deg_2)
        dr2 = dr_at_probing_line(xyz2, deltar2)
        return (dr1 - dr2)**2

    # determining initial point for the stitching optimization
    refvalue = 1e+40
    zinit = 0.
    anginit = 0.
    for ang in np.linspace(init_deg_min, init_deg_max, init_deg_step):
        for z in np.linspace(init_z_min, init_z_max, init_z_step):
            ans = fun([z, ang]).sum()
            if ans < refvalue:
                refvalue = ans
                zinit = z
                anginit = ang
    # stitching optimization
    bounds = [[opt_var_z_min+zinit, opt_var_deg_min+anginit],
              [opt_var_z_max+zinit, opt_var_deg_max+anginit]]
    res = least_squares(fun, x0=[zinit, anginit], bounds=bounds, **ls_kwargs)
    assert res.success
    xyz2, deltar1 = get_xyz_imp(res.x, bf2, pos_deg=pos_deg_2)
    dr2 = dr_at_probing_line(xyz2, deltar1)

    out = dict(delta_deg=res.x[1], delta_z=res.x[0], dr1=dr1, dr2=dr2)

    return out

    #plot_lines = False
    #if plot_lines:
        #import matplotlib
        #matplotlib.use('TkAgg')
        #import matplotlib.pyplot as plt
        #plt.clf()
        #plt.plot(probe_zarray, dr1, 'k')
        #xyz2, dr2 = get_xyz_imp(opt[-1], bf2, deg2rad([-45, +45]), angles[j])
        #dr2 = dr_at_probing_line(xyz2, dr2)
        #plt.plot(probe_zarray, dr2)
        #plt.savefig(fname=cyl_mea + '-' + str(i) + '-' + str(j) + '.png',
                #bbox_inches='tight')
