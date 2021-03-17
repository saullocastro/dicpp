import sys
sys.path.append('..')

import os
from io import BytesIO
from zipfile import ZipFile
import requests

import numpy as np

from dicpp.fit_data import best_fit_cylinder
from dicpp.fit_data import calc_Rx, calc_Ry, calc_Rz
from dicpp.stitch import stitch

def extract_file_urlzip(file_url, file_name):
    url = requests.get(file_url)
    zipfile = ZipFile(BytesIO(url.content))
    success = False
    for fname in zipfile.namelist():
        if file_name in fname:
            zipfile.extract(fname, '.')
            success = True
    assert success

def test_stitch(plot=False):
    file_url = r'https://zenodo.org/record/4608398/files/S1-1-measurement-1.zip?download=1'

    Href = 300
    H = Href - 2*(25) #TODO check this 30
    R = 136/2

    clip_box_dict = {'S1-1-1': [-1000000.0, 1000000.0, -130, 120, -1285.8867388793149, -1263.97], 'S1-1-2': [-1000000.0, 1000000.0, -130, 120, -1293.836738879315, -1271.92], 'S1-1-3': [-1000000.0, 1000000.0, -130, 120, -1293.5167388793147, -1271.6], 'S1-2-1': [-1000000.0, 1000000.0, -130, 120, -1299.1567388793148, -1277.24], 'S1-2-2': [-1000000.0, 1000000.0, -130, 120, -1290.4967388793148, -1268.58], 'S1-2-3': [-1000000.0, 1000000.0, -130, 120, -1288.9467388793148, -1267.03], 'S1-3-1': [-1000000.0, 1000000.0, -130, 120, -1289.1667388793148, -1267.25], 'S1-3-2': [-1000000.0, 1000000.0, -130, 120, -1287.2067388793148, -1265.29], 'S1-3-3': [-1000000.0, 1000000.0, -130, 120, -1287.9967388793148, -1266.08], 'S2-1-1': [-1000000.0, 1000000.0, -130, 120, -1290.7767388793147, -1268.86], 'S2-1-2': [-1000000.0, 1000000.0, -130, 120, -1304.7167388793148, -1282.8], 'S2-1-3': [-1000000.0, 1000000.0, -130, 120, -1289.3667388793149, -1267.45], 'S2-2-1': [-1000000.0, 1000000.0, -130, 120, -1284.8967388793149, -1262.98], 'S2-2-2': [-1000000.0, 1000000.0, -130, 120, -1290.8667388793149, -1268.95], 'S2-2-3': [-1000000.0, 1000000.0, -130, 120, -1293.2667388793147, -1271.35], 'S2-3-1': [-1000000.0, 1000000.0, -130, 120, -1281.6867388793148, -1259.77], 'S2-3-2': [-1000000.0, 1000000.0, -130, 120, -1292.4367388793148, -1270.52], 'S2-3-3': [-1000000.0, 1000000.0, -130, 120, -1291.0167388793147, -1269.1], 'S4-1-1': [-1000000.0, 1000000.0, -130, 120, -1288.056738879315, -1266.14], 'S4-1-2': [-1000000.0, 1000000.0, -130, 120, -1279.1567388793148, -1257.24], 'S4-1-3': [-1000000.0, 1000000.0, -130, 120, -1290.316738879315, -1268.4], 'S4-2-1': [-1000000.0, 1000000.0, -130, 120, -1325.086738879315, -1303.17], 'S4-2-2': [-1000000.0, 1000000.0, -130, 120, -1322.4467388793148, -1300.53], 'S4-2-3': [-1000000.0, 1000000.0, -130, 120, -1320.346738879315, -1298.43], 'S4-3-1': [-1000000.0, 1000000.0, -130, 120, -1276.0067388793148, -1254.09], 'S4-3-2': [-1000000.0, 1000000.0, -130, 120, -1285.796738879315, -1263.88], 'S4-3-3': [-1000000.0, 1000000.0, -130, 120, -1294.0367388793147, -1272.12], 'S8-1-1': [-1000000.0, 1000000.0, -130, 120, -1293.2267388793148, -1271.31], 'S8-1-2': [-1000000.0, 1000000.0, -130, 120, -1278.2067388793148, -1256.29], 'S8-1-3': [-1000000.0, 1000000.0, -130, 120, -1284.2667388793147, -1262.35], 'S8-2-1': [-1000000.0, 1000000.0, -130, 120, -1314.076738879315, -1292.16], 'S8-2-2': [-1000000.0, 1000000.0, -130, 120, -1322.3967388793149, -1300.48], 'S8-2-3': [-1000000.0, 1000000.0, -130, 120, -1319.4867388793148, -1297.57], 'S8-3-1': [-1000000.0, 1000000.0, -130, 120, -1292.1367388793149, -1270.22], 'S8-3-2': [-1000000.0, 1000000.0, -130, 120, -1295.576738879315, -1273.66], 'S8-3-3': [-1000000.0, 1000000.0, -130, 120, -1296.9667388793148, -1275.05]}

    x = 8
    y = -160.3
    z = 1319
    cylinder = 'S1-1'
    measurement = '1'
    ang1 = '300'
    ang2 = '240'
    fname1 = '{0}-measurement-{1}-{2}deg.csv'.format(cylinder, measurement, ang1)
    fname2 = '{0}-measurement-{1}-{2}deg.csv'.format(cylinder, measurement, ang2)
    name1 = os.path.join('.', '{0}-measurement-{1}'.format(cylinder, measurement), fname1)
    name2 = os.path.join('.', '{0}-measurement-{1}'.format(cylinder, measurement), fname2)
    extract_file_urlzip(file_url, fname1)
    extract_file_urlzip(file_url, fname2)

    clip_box = clip_box_dict[cylinder + '-' + measurement]

    kwargs = dict(ls_kwargs=dict(ftol=None, xtol=1e-6, gtol=None, method='trf',
            max_nfev=1000, jac='3-point', tr_solver='exact'))
    bf1 = best_fit_cylinder(name1, H, R_expected=R,
            best_fit_with_fixed_radius=True,
            output_path='bf.pickle', clip_box=clip_box, x0=x,
            y0=y, z0=z, alpha0=np.pi/2, beta0=-0.01,
            loadtxt_kwargs=dict(delimiter=',', usecols=(2, 3, 4), skiprows=1),
            **kwargs)
    bf2 = best_fit_cylinder(name2, H, R_expected=R,
            best_fit_with_fixed_radius=True,
            output_path='bf.pickle', clip_box=clip_box, x0=x,
            y0=y, z0=z, alpha0=np.pi/2, beta0=-0.01,
            loadtxt_kwargs=dict(delimiter=',', usecols=(2, 3, 4), skiprows=1),
            **kwargs)

    line_len = 1000
    Hcut = 50
    probe_zarray = np.linspace(Hcut, Hcut+(Href-2*Hcut), line_len, endpoint=True)
    out = stitch(bf1, bf2,
        pos_deg1=+90,
        pos_deg2=+90+60,
        height_ref=Href,
        probe_deg=30, probe_dist_deg=10, probe_R=R,
        probe_zarray=probe_zarray, init_deg_min=-5, init_deg_max=+5,
        init_deg_step=11, init_z_min=-20, init_z_max=+20, init_z_step=11,
        opt_var_z_min=-10., opt_var_z_max=+10., opt_var_deg_min=-5.,
        opt_var_deg_max=+5.)

    if plot:
        import pyvista as pv
        from pyvista import colors
        plotter = pv.Plotter()

        xyz = bf1['input_pts'].T
        x0 = bf1['x0']
        y0 = bf1['y0']
        z0 = bf1['z0']
        z1 = bf1['z1']
        alpharad = bf1['alpharad']
        betarad = bf1['betarad']
        Rx = calc_Rx(alpharad)
        Ry = calc_Ry(betarad)
        xyz[:, 0] += x0
        xyz[:, 1] += y0
        xyz[:, 2] += z0
        xyz = ((Ry @ Rx) @ xyz.T).T
        xyz[:, 2] += z1
        Rz = calc_Rz(np.deg2rad(90))
        xyz = (Rz @ xyz.T).T
        points = pv.PolyData(xyz)
        plotter.add_mesh(points, render_points_as_spheres=True,
                color=colors.hexcolors['gray'], smooth_shading=True, point_size=6.)

        xyz = bf2['input_pts'].T
        x0 = bf2['x0']
        y0 = bf2['y0']
        z0 = bf2['z0']
        z1 = bf2['z1']
        alpharad = bf2['alpharad']
        betarad = bf2['betarad']
        Rx = calc_Rx(alpharad)
        Ry = calc_Ry(betarad)
        xyz[:, 0] += x0
        xyz[:, 1] += y0
        xyz[:, 2] += z0
        xyz = ((Ry @ Rx) @ xyz.T).T
        xyz[:, 2] += z1 + out['delta_z']
        Rz = calc_Rz(np.deg2rad(90 + 60 + out['delta_deg']))
        xyz = (Rz @ xyz.T).T
        points = pv.PolyData(xyz)
        plotter.add_mesh(points, render_points_as_spheres=True,
                color=colors.hexcolors['gray'], smooth_shading=True, point_size=6.)

        points = pv.PolyData(xyz)
        plotter.add_mesh(points, render_points_as_spheres=True,
                color=colors.hexcolors['lightblue'], smooth_shading=True, point_size=6.)
        xaxis = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=70)
        plotter.add_mesh(xaxis, color='black')
        yaxis = pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), scale=70)
        plotter.add_mesh(yaxis, color='black')
        zaxis = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=70)
        plotter.add_mesh(zaxis, color='black')
        plotter.set_background('white')
        plotter.show()


if __name__ == '__main__':
    test_stitch(plot=True)
