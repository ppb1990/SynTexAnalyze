import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from numpy import sin, cos,arcsin,arccos,arctan
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D





def to_xyz(Phi, Chi, R):
    x = [R * sin(np.radians(x)) * sin(np.radians(y)) for x, y in zip(Phi, Chi)]
    y = [R * cos(np.radians(x)) * sin(np.radians(y)) for x, y in zip(Phi, Chi)]
    z = [R * cos(np.radians(y)) for y in Chi]
    return x, y, z

def azmR_to_xyz(azm, r, R=1, theta=0,  operando=False):
    # Need to rotate the Phi, Chi
    # Phi is the azimuth angle
    # Chi is the rotation angle
    # Need to calculate the corresponding Phi and Chi at r=0
    # first, for each r, assume rotation is 0, and calculate the X,Y,Z
    # then, for each X,Y,Z, rotate it based on the rotation angle

    '''
    dc = {'x': [], 'y': [], 'z': []}
    Phi = [i - 90 for i in azm]  # shift 90 degrees
    Chi = [90 - theta for i in r]  # initial value the same
    tx, ty, tz = to_xyz(Phi, Chi, R)  # calculate the corresponding x,y,z based on the initial Chi

    for dx, dy, dz, angle in zip(tx,ty,tz,r):
        rotation = Rotation.from_euler('y', angle, degrees=True)
        temp_x,temp_y,temp_z = rotation.apply([dx, dy, dz])
        for key, value in zip(['x', 'y', 'z'], [temp_x, temp_y, temp_z]):
            dc[key].append(value)
    '''

    dc = {'x': [], 'y': [], 'z': []}
    Phi = [i - 90 for i in azm]  # shift 90 degrees
    Chi = [90 - theta for i in r]  # initial value the same
    tx, ty, tz = to_xyz(Phi, Chi, R)  # calculate the corresponding x,y,z based on the initial Chi

    # under the operando conditions, the coordinate will be rotated along 'y' direction for 90 degrees,
    # and then rotated along z directions depends on the angle
    if operando:
        rotation = Rotation.from_euler('y', 90, degrees=True)
        for dx, dy, dz in zip(tx, ty, tz):
            temp_x, temp_y, temp_z = rotation.apply([dx, dy, dz])
            for key, value in zip(['x', 'y', 'z'], [temp_x, temp_y, temp_z]):
                dc[key].append(value)
        tx = [i for i in dc['x']]  # reassign the value
        ty = [i for i in dc['y']]
        tz = [i for i in dc['z']]
        dc = {'x': [], 'y': [], 'z': []}  # clean the dc

    for dx, dy, dz, angle in zip(tx, ty, tz, r):
        if operando:
            rotation = Rotation.from_euler('z', angle, degrees=True)
        elif not operando:
            rotation = Rotation.from_euler('y', angle, degrees=True)
        temp_x, temp_y, temp_z = rotation.apply([dx, dy, dz])
        for key, value in zip(['x', 'y', 'z'], [temp_x, temp_y, temp_z]):
            dc[key].append(value)

    return dc['x'], dc['y'], dc['z']

def rotation_c4(x, y, z):
    cord_lst = [[x, y, z]]
    for i in range(1,4):
        r = Rotation.from_euler('z', angles=i * 90, degrees=True)
        temp_x, temp_y, temp_z = r.apply([x, y, z])
        cord_lst.append([temp_x, temp_y, temp_z])
    return cord_lst

def rotation_110(x,y,z):
    # !!!!!!!!
    # all need to be changed that the rotate axis should be relative to the diffraction spot
    # instead of the physical axis
    # !!!!!!!!
    # total of 12 equivalents
    # 1. rotate along z for 90, 180, 270 c-clock
    #r1 = Rotation.from_euler('z', 90, degrees=True)
    # 2. rotate along another x for 45 c-clock, and then repeat #1 operation
    #r2 = Rotation.from_euler('x', 45, degrees=True)
    #r3 = Rotation.from_euler('z', 45, degrees=True)
    # 3. inversion operation (-x, -y, -z)

    # assume the spot we found is 110
    # then need to calculate the x, y, z
    # and rotate along those axes
    cord_lst = [[x,y,z]]
    # 1
    for i in range(1,4):
        r = Rotation.from_euler('z', angles=i*90, degrees=True)
        temp_x,temp_y,temp_z = r.apply([x,y,z])
        cord_lst.append([temp_x, temp_y, temp_z])
    # 2
    r1 = Rotation.from_euler('x', 45, degrees=True)
    r2 = Rotation.from_euler('z', 45, degrees=True)
    x1, y1, z1 = r1.apply([x,y,z])
    x2, y2, z2 = r2.apply([x1,y1,z1]) # get the off plane one
    cord_lst.append([x2,y2,z2])
    for i in range(1,4):
        r = Rotation.from_euler('z', angles=i*90, degrees=True)
        temp_x,temp_y,temp_z = r.apply([x2,y2,z2])
        cord_lst.append([temp_x, temp_y, temp_z])

    # 3
    for i in range(0, len(cord_lst)):
        temp_lst = [-j for j in cord_lst[i]]
        if temp_lst not in cord_lst:
            cord_lst.append(temp_lst)

    return cord_lst

def rotation_200(x, y, z):
    # total of 6 equivalents
    # 1. rotate along x for 90, 180, 270
    # 2. rotate along y for 90, 180, 270
    # 3. rotate along z for 90, 180, 270

    # for now, just do a c4 rotation

    cord_lst = [[x, y, z]]
    for i in range(1, 4):
        r = Rotation.from_euler('z', angles=i * 90, degrees=True)
        temp_x, temp_y, temp_z = r.apply([x, y, z])
        cord_lst.append([temp_x, temp_y, temp_z])

    return cord_lst

def rotation_211(x, y, z):
    # total of 24 equivalents
    # 1. rotate along (1, 1, 1)
    # 2. rotate along z for 90, 180, 270
    # 3. inversion

    # for now, just do a c4 rotation

    cord_lst = [[x, y, z]]
    for i in range(1, 4):
        r = Rotation.from_euler('z', angles=i * 90, degrees=True)
        temp_x, temp_y, temp_z = r.apply([x, y, z])
        cord_lst.append([temp_x, temp_y, temp_z])
    return cord_lst


def get_inverse(x, y, z, intensity, add=True, carrier=None):
    if carrier:  # carry other information with the operation
        # Try to convert carrier values to lists if not already lists
        for key, value in carrier.items():
            if not isinstance(value, list):
                try:
                    carrier[key] = list(value)  # Convert to list
                except TypeError:
                    raise ValueError(f"The value for key '{key}' in carrier cannot be converted to a list.")

        if len(x) != len(list(carrier.values())[0]):
            raise ValueError('The length of items within carrier does not match the length of x.')

        # Ensure intensity is a list
        if not isinstance(intensity, list):
            intensity = list(intensity)

        dc = {'x': [], 'y': [], 'z': [], 'intensity': []}
        original_dc = {'x': x, 'y': y, 'z': z, 'intensity': intensity}
        for key, value in carrier.items():
            dc[key] = []
            original_dc[key] = value

        for i in range(len(x)):
            values = {key: original_dc[key][i] for key in original_dc.keys()}
            for key, value in values.items():
                if key in ['x', 'y', 'z']:
                    dc[key].append(-value)
                else:  # remain the same value for items other than x, y, z
                    dc[key].append(value)

        if add:  # Append original values to the mirrored values
            for key in original_dc.keys():
                dc[key].extend(original_dc[key])

        return dc['x'], dc['y'], dc['z'], dc['intensity'], dc
        # return the full dc for now, might change the function later to just return dc for both conditions
    else:
        dc = {'x': [], 'y': [], 'z': [], 'intensity': []}
        if not isinstance(intensity, list):
            intensity = list(intensity)
        for dx, dy, dz, di in zip(x, y, z, intensity):
            values = [dx, dy, dz, di]
            for key, value in zip(dc.keys(), values):
                if key == 'intensity':
                    dc[key].append(value)  # add intensity twice with the same value
                else:
                    dc[key].append(-value)  # add the mirrored value

        if add:  # add the mirror to the original file
            for key, value in zip(dc.keys(), [x, y, z, intensity]):
                dc[key] = dc[key] + value

        return dc['x'], dc['y'], dc['z'], dc['intensity']


def get_symmetry(x, y, z, intensity, operation='110'):
    dc = {'x': [], 'y': [], 'z': [], 'intensity': []}
    if not isinstance(intensity, list):
        intensity = list(intensity)
    for dx, dy, dz, di in zip(x, y, z, intensity):
        if operation == '110':
            cord_lst = rotation_110(dx, dy, dz)
        elif operation == '200':
            cord_lst = rotation_200(dx, dy, dz)
        elif operation == '211':
            cord_lst = rotation_200(dx, dy, dz)
        elif operation == 'c4':
            cord_lst = rotation_c4(dx, dy, dz)

        for cords in cord_lst:
            for key, value in zip(dc.keys(),cords + [di]):
                dc[key].append(value)
    return dc['x'], dc['y'], dc['z'], dc['intensity']


def get_sphere(x, y, z, intensity, sphere='upper', carrier=None):
    if carrier:  # carry other information with the operation
        # Try to convert carrier values to lists if not already lists
        for key, value in carrier.items():
            if not isinstance(value, list):
                try:
                    carrier[key] = list(value)  # Convert to list
                except TypeError:
                    raise ValueError(f"The value for key '{key}' in carrier cannot be converted to a list.")

        if len(x) != len(list(carrier.values())[0]):
            raise ValueError('The length of items within carrier does not match the length of x.')

        # Ensure intensity is a list
        if not isinstance(intensity, list):
            intensity = list(intensity)

        dc = {'x': [], 'y': [], 'z': [], 'intensity': []}
        original_dc = {'x': x, 'y': y, 'z': z, 'intensity': intensity}
        for key, value in carrier.items():
            dc[key] = []
            original_dc[key] = value

        for i in range(len(x)):
            values = {key: original_dc[key][i] for key in original_dc.keys()}
            dz = original_dc['z'][i]
            if sphere == 'upper':
                if dz >= 0:
                    for key, value in zip(original_dc.keys(), values):
                        dc[key].append(value)
            elif sphere == 'lower':
                if dz <= 0:
                    for key, value in zip(original_dc.keys(), values):
                        dc[key].append(value)

        return dc['x'], dc['y'], dc['z'], dc['intensity'], dc
        # return the full dc for now, might change the function later to just return dc for both conditions
    else:
        dc = {'x': [], 'y': [], 'z': [], 'intensity': []}
        if not isinstance(intensity, list):
            intensity = list(intensity)
        for dx, dy, dz, di in zip(x, y, z, intensity):
            values = [dx, dy, dz, di]

            if sphere == 'upper':
                if dz >= 0:
                    for key, value in zip(dc.keys(), values):
                        dc[key].append(value)
            elif sphere == 'lower':
                if dz <= 0:
                    for key, value in zip(dc.keys(), values):
                        dc[key].append(value)

        return dc['x'], dc['y'], dc['z'], dc['intensity']

def read_spots(file_path, system = 'mac',offset=None,get_tth=False):
    if offset is None:
        offset = 100  # for the XPD 2024-2 cycle, the initial angle is 100
    file_df = pd.read_csv(file_path)
    if system == 'mac':  # depends on the system
        f_name = file_path.split('_total_')[0].split('/')[-1]
    elif system == 'windows':
        f_name = file_path.split('_total_')[0].split('\\')[-1]
    count = file_path.split('_total_')[1].split('.')[0]
    r = [i - offset for i in file_df['rotation']]  # 100 is the offset f
    azm = file_df['azm']
    intensity = file_df['intensity']
    if get_tth:
        tth = file_df['tth']
        return f_name, count, azm, r, intensity,tth
    else:
        return f_name, count, azm, r, intensity

def get_Ewarld():
    pass

def Ewald_basic(qv_length):  # fig need to be an plt.figure() object
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')

    # draw a surface
    xx, yy = np.meshgrid([-1, 1], [-1, 1])
    z = xx * 0  # x-y plane
    ax.plot_surface(xx, yy, z, alpha=0.5, color='gray')

    # draw a sphere
    u, v = np.mgrid[0:2 * np.pi:360j, 0:np.pi:360j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.5)

    # draw a point at the center
    ax.scatter([0], [0], [0], color="g", s=100)

    # draw arrows for x, y, z
    ax.quiver(0, 0, 0, 1, 0, 0, length=qv_length, arrow_length_ratio=0.05, color='r')
    ax.quiver(0, 0, 0, 0, 1, 0, length=qv_length*1.5, arrow_length_ratio=0.05, color='r')
    ax.quiver(0, 0, 0, 0, 0, 1, length=qv_length, arrow_length_ratio=0.05, color='r')

    return fig, ax

def Ewald_add_scatter(ax, azm, r, intensity, theta, **kwargs):
    if 'operando' in kwargs.keys():  # operando measurement
        X, Y, Z = azmR_to_xyz(azm, r, R=1, theta=theta, operando=True)
    else:  # default ex-situ measurement
        X, Y, Z = azmR_to_xyz(azm, r, R=1, theta=theta)
    area = intensity / np.min(intensity)
    ax.scatter(X, Y, Z, s=area, cmap='gray', c=intensity, alpha=0.5)

def Ewald_get_projection(file_path, theta,
                         gv_length=1.5, inverse=True, show_sphere=True, sphere='upper', **kwargs):
    #fig = plt.figure(figsize=(10, 10))
    fig, ax = Ewald_basic(qv_length=gv_length)
    name, count, azm, r, intensity = read_spots(file_path)
    if 'operando' in kwargs.keys():  # operando measurement
        X, Y, Z = azmR_to_xyz(azm, r, R=1, theta=theta, operando=True)
    else:  # default ex-situ measurement
        X, Y, Z = azmR_to_xyz(azm, r, R=1, theta=theta)
    if inverse:
        X, Y, Z, azm, intensity = get_inverse(X, Y, Z, azm, intensity)
    if show_sphere:
        X, Y, Z, azm, intensity = get_sphere(X, Y, Z, azm, intensity, sphere=sphere)

    area = intensity / np.min(intensity)
    ax.scatter(X, Y, Z, s=area, cmap='gray', c=intensity, alpha=0.5)
    ax.view_init(90, 0, 0)
    if 'title' in kwargs.keys() and kwargs['title'] == True:
        ax.set_title('Name: {}\nCount: {}\nSphere: {}'.format(name, count, sphere), y=0.95)
    if 'out_path' in kwargs.keys():
        out_path = kwargs['out_path']
        out_name = '{}_{}_projection.jpg'.format(name,count)
        out_path = os.path.join(out_path,out_name)
        fig.savefig(out_path)

def Ewald_get_sideview(file_path, theta,
                         gv_length=1.5, inverse=True, show_sphere=True, sphere='upper', **kwargs):
    # fig = plt.figure(figsize=(10, 10))
    fig, ax = Ewald_basic(qv_length=gv_length)
    name, count, azm, r, intensity = read_spots(file_path)
    if 'operando' in kwargs.keys():  # operando measurement
        X, Y, Z = azmR_to_xyz(azm, r, R=1, theta=theta, operando=True)
    else:  # default ex-situ measurement
        X, Y, Z = azmR_to_xyz(azm, r, R=1, theta=theta)
    if inverse:
        X, Y, Z, azm, intensity = get_inverse(X, Y, Z, azm, intensity)
    if show_sphere:
        X, Y, Z, azm, intensity = get_sphere(X, Y, Z, azm, intensity, sphere=sphere)

    area = intensity / np.min(intensity)
    ax.scatter(X, Y, Z, s=area, cmap='gray', c=intensity, alpha=0.5)
    ax.view_init(0, 90, 0)
    if 'title' in kwargs.keys() and kwargs['title'] == True:
        ax.set_title('Name: {}\nCount: {}\nSphere: {}'.format(name, count, sphere), y=0.95)
    if 'out_path' in kwargs.keys():
        out_path = kwargs['out_path']
        out_name = '{}_{}_SideView.jpg'.format(name,count)
        out_path = os.path.join(out_path, out_name)
        fig.savefig(out_path)










