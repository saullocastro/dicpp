
import sys
import pickle

import numpy as np
from numpy import deg2rad
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from dicpp.fit_data import calc_Rx, calc_Ry, calc_Rz, calc_radius_ellipse
from dicpp.interpolate import inv_weighted


def stitch_cylinder(diff_step_deg=0.2):
    """
    Parameters
    ----------
    diff_step_deg : float, optional
        Angular step size used in the finite difference to compute gradients

    """
    Href = 300
    Hcut = 50
    H = Href - 2*Hcut
    R = 136/2

    line_len = 1000
    zline = np.linspace(Hcut, Hcut+H, line_len, endpoint=True)


    def get_xyz_imp(p, name, minmax, nominal_pos_deg):
        off_deg = p[0]
        zoff = p[1]
        off = deg2rad(off_deg)
        with open(name, 'rb') as f:
            ans = pickle.load(f)
        Rx = calc_Rx(ans['alpharad'])
        Ry = calc_Ry(ans['betarad'])
        Rz = calc_Rz(ans['gammarad'])
        z_offset = ans['z_offset']
        x0 = ans['x0']
        y0 = ans['y0']
        z0 = ans['z0']
        xyz = ans['input_pts'].T
        xyz[:, 0] += x0
        xyz[:, 1] += y0
        xyz[:, 2] += z0
        xyz = ((Rz @ Ry @ Rx ) @ xyz.T).T
        xyz[:, 2] += z_offset

        xyz[:, 2] -= xyz[:, 2].mean()
        xyz[:, 2] += Href/2

        a = ans['a_best_fit']
        b = ans['b_best_fit']
        ang1, ang2 = minmax
        theta = np.arctan2(xyz[:, 1], xyz[:, 0])
        radius_ellipse = calc_radius_ellipse(a, b, theta)
        deltar = (xyz[:, 0]**2 + xyz[:, 1]**2)**0.5 - radius_ellipse
        rcnew = R + deltar
        xyz[:, 0] = rcnew*np.cos(theta)
        xyz[:, 1] = rcnew*np.sin(theta)
        xyz[:, 2] += zoff
        #NOTE with the following rotation, position 1 is perpendicular to the
        #     x-axis, already considering the offset
        Rz = calc_Rz(off + np.pi/2)
        xyz = (Rz @ xyz.T).T
        #trimming
        theta = np.arctan2(xyz[:, 1], xyz[:, 0])
        valid = (theta >= -ang1) & (theta <= +ang2)
        theta = theta[valid]
        xyz = xyz[valid]
        deltar = deltar[valid]
        #rotating face to nominal circumferential position
        Rz = calc_Rz(deg2rad(-nominal_pos_deg))
        xyz = (Rz @ xyz.T).T
        return xyz, deltar

    def get_line(theta_deg, xyz, imp):
        xyz_mesh = np.zeros((zline.shape[0], 3))
        xyz_mesh[:, 0] = R*np.cos(deg2rad(theta_deg))
        xyz_mesh[:, 1] = R*np.sin(deg2rad(theta_deg))
        xyz_mesh[:, 2] = zline
        dist, out = inv_weighted(imp, xyz, xyz_mesh, ncp=10,
                power_parameter=1.7)
        return out


    xyz1, imp1 = get_xyz_imp([0, 0], names[i], deg2rad([-45, +45]), angles[i])
    out1 = get_line(theta_deg, xyz1, imp1)

    def fun(p):
        xyz2, imp2 = get_xyz_imp(p, names[j], minmax[j], angles[j])
        out2 = get_line(theta_deg, xyz2, imp2)
        return (out1 - out2)**2

    refvalue = 1e+40
    zinit = 0.
    anginit = 0.
    for ang in np.linspace(-10, +10, 11):
        for z in np.linspace(-20., +20., 11):
            ans = fun([z, ang]).sum()
            if ans < refvalue:
                refvalue = ans
                zinit = z
                anginit = ang
    bounds = [[-10.+zinit, -10.+anginit],
              [+10.+zinit, +10.+anginit]]
    res = least_squares(fun, x0=[zinit, anginit], xtol=1e-4, bounds=bounds,
            diff_step=diff_step_deg, jac='3-point', ftol=None, gtol=None)
    assert res.success

    opt.append(res.x)

    plot_lines = False
    if plot_lines:
        plt.clf()
        plt.plot(zline, out1, 'k')
        xyz2, imp2 = get_xyz_imp(opt[-1], names[j], minmax[j], angles[j])
        out2 = get_line(theta_deg, xyz2, imp2)
        plt.plot(zline, out2)
        plt.savefig(fname=cyl_mea + '-' + str(i) + '-' + str(j) + '.png',
                bbox_inches='tight')

    return opt







