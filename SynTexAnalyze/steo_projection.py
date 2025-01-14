import os
import numpy as np
from matplotlib import pyplot as plt

from numpy import sqrt


def to_SteoChi(z, R=1 , sphere='upper', angle=True):
    # only z is used to determine the Chi angle

    if sphere == 'upper':
        Chi = [R * sqrt(R**2 - i**2) / (R+i) for i in z]

    if sphere == 'lower':
        Chi = [R * sqrt(R**2 - i**2) / (R+abs(i)) for i in z]
    if angle:
        Chi = [i * 90 / R for i in Chi]

    return Chi
def to_SteoPhi(x, y, angle=True, to_list=True):
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    phase_x = x/abs(x)
    phase_y = y/abs(y)
    phase_idx = abs(2 * phase_x - phase_y - abs(phase_y))
    # print(phase_x, phase_y, phase_idx)
    # return value [0,2,2,4] for [upper_right, lower_right, lower_left, upper_left]
    if angle:
        SteoPhi = np.arctan(x/y)
        SteoPhi = np.rad2deg(SteoPhi) + phase_idx * 90
    else:
        SteoPhi = np.arctan(x / y) + np.deg2rad(phase_idx * 90)
    if to_list:
        SteoPhi = [i for i in SteoPhi]
    return SteoPhi


def get_SteoPhiChi(x, y, z, R=1, sphere='upper',angle=True):
    steoPhi = to_SteoPhi(x, y, angle=angle)
    steoChi = to_SteoChi(z, R=R, sphere=sphere ,angle=angle)

    return steoPhi, steoChi

def Steo_basic():
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_rorigin(0)
    ax.set_theta_zero_location('W',offset=0)
    plt.thetagrids((0,90,180,270))
    #ax.set_thetagradis((0,90,180,270))
    return fig, ax

def Steo_add_scatter(ax,SteoPhi, SteoChi, intensity, show_area=True):

    if show_area:
        area = intensity / np.min(intensity)
        ax.scatter(SteoPhi, SteoChi, s=area, cmap='gray', c=intensity, alpha=0.5)
    else:
        ax.scatter(SteoPhi, SteoChi, cmap='gray', c=intensity, alpha=0.5)

def Steo_get_projection(SteoPhi, SteoChi, intensity, name, count, sphere, show_area=True, **kwargs):
    fig, ax = Steo_basic()

    if 'log10' in kwargs and kwargs['log10']:  # log scale
        intensity = [np.log10(i+1) for i in intensity]

    if show_area:
        Steo_add_scatter(ax, SteoPhi, SteoChi, intensity)
    else:
        Steo_add_scatter(ax, SteoPhi, SteoChi, intensity, show_area=False)
    # add the scale bar here
    if 'title' in kwargs.keys() and kwargs['title']:
        ax.set_title('{}\nCount: {}\nSphere: {}'.format(name, count, sphere))  # , y=0.95)
    if 'out_path' in kwargs.keys():
        out_path = kwargs['out_path']
        out_name = '{}_{}_SteoProjection.jpg'.format(name,count)
        out_path = os.path.join(out_path, out_name)
        fig.savefig(out_path)
    return fig, ax

def Steo_get_density(ds_Phi, ds_Chi, SteoPhi, SteoChi, **kwargs):
    # use np.histogram2d to calculate the density
    from numpy import histogram2d as h2d
    if 'intensity' in kwargs.keys():
        intensity = kwargs['intensity']
        ds_intensity, *_ = h2d(SteoPhi, SteoChi, bins=[ds_Phi,ds_Chi], weights=intensity)
        # ds_intensity, *_ = h2d(SteoPhi, SteoChi, bins=(3,3),weights=intensity)
    else:
        ds_intensity, *_ = h2d(SteoPhi, SteoChi, bins=[ds_Phi, ds_Chi])
    return ds_intensity

def Steo_get_heatmap(angle_min, angle_max, step_size, R=1, resolution=3, theta=2.1, **kwargs):

    from numpy import sin, cos
    from scipy.spatial.transform import Rotation
    import intensity_to_Ewald as IE
    def to_xyz(R, Phi, Chi):
        x = [R * sin(x) * sin(y) for x in Phi for y in Chi]
        y = [R * cos(x) * sin(y) for x in Phi for y in Chi]
        z = [R * cos(y) for x in Phi for y in Chi]
        return x, y, z
    def tilt(count,angle):  # intensity correction for tilting
        return count * (1/np.cos(np.deg2rad(abs(angle))))

    def rotate(dc, X, Y, Z, count, angle_min, angle_max, step_size):
        steps = int(abs(angle_max-angle_min)/step_size) + 1

        for i in range(0, steps+1):
            angle = angle_min + step_size * i
            if angle != 0:  # ignore the original
                for ix, iy, iz, ii in zip(X, Y, Z, count):
                    matrix = [ix, iy, iz]
                    r = Rotation.from_euler('y', angle, degrees=True)
                    temp_x, temp_y, temp_z = r.apply(matrix)
                    temp_count = tilt(ii,angle)

                    for key, value in zip(['x', 'y', 'z'], [temp_x, temp_y, temp_z]):
                        dc[key].append(value)
                    dc['count'].append(temp_count)



    Phi_steps = int(180 / resolution + 1)  # should be the same as the resolution
    Phi = np.radians(np.linspace(0, 180, Phi_steps))
    Chi = np.radians([90-theta])

    X, Y, Z = to_xyz(R, Phi, Chi)
    dc = {'x': [], 'y': [], 'z': [], 'count': []}
    for i, j, k, in zip(X, Y, Z):
        temp_x, temp_y, temp_z = [i, j, k]
        for key, value in zip(['x', 'y', 'z'], [temp_x, temp_y, temp_z]):
            dc[key].append(value)
    count = [1 for i in X]
    dc['count'] = count

    rotate(dc, X, Y, Z, count, angle_min, angle_max, step_size)

    X2, Y2, Z2, count2 = IE.get_inverse(dc['x'], dc['y'], dc['z'], dc['count'])
    X3, Y3, Z3, count3 = IE.get_sphere(X2, Y2, Z2, count2)

    if 'symmetry_operation' in kwargs.keys():# add the symmetry operation
        if kwargs['symmetry_operation'] == 'c4':
            pass
    else:
        Phi, Chi, total_count = Steo_get_contour(X3, Y3, Z3, count3, '',0, resolution=resolution, log=False,get='data', zero_to_nan=True)

    return Phi, Chi, total_count


def Steo_get_contour(x, y, z, intensity, name, count,
                     resolution=3,
                     sphere='upper',
                     log=True,
                     get='figure',
                     scale=1000,
                     get_array=False,
                     HeatMap=None,
                     **kwargs):

    from orix.sampling.S2_sampling import _sample_S2_equal_area_coordinates as get_equal_crd
    from matplotlib import cm

    # if 'log10' in kwargs and kwargs['log10']:  # log scale
    #     intensity = [np.log10(i+1) for i in intensity]

    ds_Phi, ds_Chi = get_equal_crd(resolution=resolution, hemisphere=sphere, azimuth_endpoint=True)
    # shift Phi by half of the bin to make it center
    delta_Phi = (ds_Phi[1] - ds_Phi[0])/2
    ds_Phi = ds_Phi - delta_Phi

    ds_Phi = np.rad2deg(ds_Phi)
    ds_Chi = np.rad2deg(ds_Chi)
    Chi= to_SteoChi(z)
    Phi = to_SteoPhi(x, y)
    ds_intensity = Steo_get_density(ds_Phi, ds_Chi, Phi, Chi, intensity=intensity)

    temp = []
    for i in ds_intensity:
        temp.append([j for j in i])
    temp.append(temp[0])
    ds_intensity = np.array(temp)

    if 'zero_to_nan' in kwargs.keys() and kwargs['zero_to_nan']:
        ds_intensity[ds_intensity == 0] = np.nan  # remove all the region with no intensity
        # ds_intensity = np.log10(ds_intensity * scale + 10)

    if HeatMap != None:  # heatmap is a list of angle_min, angle_max, step_size, theta e.g. [-80,80,-5, theta]
        angle_min, angle_max, step_size, theta = [i for i in HeatMap]
        _, _, correction = Steo_get_heatmap(angle_min, angle_max, step_size, resolution=resolution, theta=theta)
        ds_intensity = ds_intensity / correction
        ds_intensity = np.nan_to_num(ds_intensity)


    msh_Phi, msh_Chi = np.meshgrid(ds_Phi, ds_Chi[:-1], indexing='ij')
    msh_Phi = np.deg2rad(msh_Phi)

    if log:
        ds_intensity = np.log10(ds_intensity * scale + 10)
    #     if 'zero_to_nan' in kwargs.keys() and kwargs['zero_to_nan']:
    #         ds_intensity[ds_intensity == 0] = np.nan  # remove all the region with no intensity
    #     if 'zero_to_log' in kwargs.keys() and kwargs['zero_to_log'] == 0:
    #         ds_intensity[ds_intensity == 0] = 1  # change it to 1 so log(1)=0
    #     elif 'zero_to_log' in kwargs.keys() and kwargs['zero_to_log'] == 1:
    #         ds_intensity[ds_intensity == 0] = 10  # change it to 1 so log(1)=0
    #     ds_intensity = np.log10(ds_intensity)  # to avoid zero in log scale
    else:
    #     if 'zero_to_nan' in kwargs.keys() and kwargs['zero_to_nan']:
    #         ds_intensity[ds_intensity == 0] = np.nan  # remove all the region with no intensity
        ds_intensity = ds_intensity * scale



    if get == 'figure':
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        ax.set_rorigin(0)
        ax.set_theta_zero_location('W', offset=0)
        plt.thetagrids((0, 90, 180, 270))
        cs = ax.contourf(msh_Phi, msh_Chi, ds_intensity, cmap = cm.RdYlBu_r)

        if 'cbar' in kwargs.keys() and kwargs['cbar']:
            cbar = plt.colorbar(cs, )

        if 'title' in kwargs.keys() and kwargs['title'] == True:
            ax.set_title('{}\nCount: {}\nSphere: {}'.format(name, count, sphere))  # , y=0.95)
        if 'out_path' in kwargs.keys():
            out_path = kwargs['out_path']
            out_name = '{}_{}_SteoDensity.jpg'.format(name,count)
            out_path = os.path.join(out_path, out_name)
            fig.savefig(out_path)
        if get_array:  # only if to get both
            return fig, ax, msh_Phi, msh_Chi, ds_intensity
        else:
            return fig, ax
    elif get == 'data':
        return msh_Phi, msh_Chi, ds_intensity


