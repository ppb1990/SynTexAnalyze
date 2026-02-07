import sys
import numpy as np
from PIL import Image
from networkx import ring_of_cliques


def d_to_tth(d, w):  # calculate the tth from d spacing
    tth = np.arcsin(w / (d * 2)) * 2  # results in radians
    tth = tth * 180 / np.pi
    return tth


def get_rotation_th(file_name):  # for texture sample, get rotation angle
    rotation_th = file_name.split('th_')[1].split(',')[0]
    return rotation_th


def get_psn_xy(file_name):  # for Austin's sample, get psn_x, psn_y
    psn_x = file_name.split('sample_x_')[1].split('mm')[0].replace(',', '.')
    psn_y = file_name.split('sample_y_')[1].split('mm')[0].replace(',', '.')
    psn_x = float(psn_x)
    psn_y = float(psn_y)
    return psn_x, psn_y


def get_tth_azm(detector, intTAmap):

    # Currenlty using numbers such as 2048, 1475, 1679 for PE and Pilatus detecors
    # In the future should use a raw file to determine the shape

    if detector == 'PE':

        im_azm = np.array([[0 for j in range(2048)] for i in range(2048)], dtype=np.float64)
        im_tth = np.array([[0 for j in range(2048)] for i in range(2048)], dtype=np.float64)

        for i in range(16):
            for j in range(16):
                temp_azm_map = intTAmap[i][j][0]
                temp_tth_map = intTAmap[i][j][1]
                for x in range(128):
                    for y in range(128):
                        out_x = 128 * i + x
                        out_y = 128 * j + y

                        im_azm[out_x, out_y] = temp_azm_map[x, y]
                        im_tth[out_x, out_y] = temp_tth_map[x, y]
    elif detector == 'Pilatus':

        # for Pilatus, the shape of the detector is very different from PE, so need to modify this

        im_azm = np.array([[0 for j in range(1475)] for i in range(1679)], dtype=np.float64)
        im_tth = np.array([[0 for j in range(1475)] for i in range(1679)], dtype=np.float64)

        for i in range(14):
            for j in range(12):
                temp_azm_map = intTAmap[i][j][0]
                temp_tth_map = intTAmap[i][j][1]

                if i == 13:  # the matrix bending is not uniform, so the last one is 15 for i and 67 for j
                    range_i = 15
                else:
                    range_i = 128
                if j == 11:
                    range_j = 67
                else:
                    range_j = 128

                for x in range(range_i):
                    for y in range(range_j):
                        out_x = 128 * i + x
                        out_y = 128 * j + y

                        im_azm[out_x, out_y] = temp_azm_map[x, y]
                        im_tth[out_x, out_y] = temp_tth_map[x, y]

    return im_tth, im_azm


def get_intTAmap(GS_path,file_path, img_ctrl_path):
    sys.path.insert(GS_path)  # needed to "find" GSAS-II modules
    import GSASIIscriptable as G2sc
    import gc

    cache = {}
    file_path = file_path[0]  # temp use
    # ctrl_path = os.path.join('D://Postdoc//XPD-2024-2//Zhuo','texture-control.imctrl')

    gpx = G2sc.G2Project(newgpx='temp.gpx')  # temp use
    img = gpx.add_image(file_path, cacheImage=False)  # don't need to cache image
    img[0].loadControls(img_ctrl_path)

    # cache['intMaskMap'] = img[0].IntMaskMap()
    cache['intTAmap'] = img[0].IntThetaAzMap()

    im_tth, im_azm = get_tth_azm(cache['intTAmap'])

    del cache, gpx

    return im_tth, im_azm


def get_search_conditions(
                              upper_lim,
                              lower_lim,
                              Li_lp_a,
                              wavelength
                          ):
    # rings should be like [0] for (110) or [0,1] for (110), (200)
    # the lim terms are used to determine the upper boundary and lower boundary
    # currently calculate all three hkls
    ring_conditions = {'hkl': ['tth', 'tth + uper_lim', 'tth - lower_lim']}

    Li_hkl = ['110', '200', '211']
    Li_hkl_sum = [2, 4, 6]  # h^2 + k^2 + l^2 for (110) (200) (211)
    Li_d = [Li_lp_a * np.sqrt(1 / i) for i in Li_hkl_sum]
    Li_tth = [d_to_tth(i, wavelength) for i in Li_d]

    if lower_lim == 'eq':
        for i, j in zip(Li_hkl, Li_tth):
            ring_conditions[i] = [j, j + upper_lim, j - upper_lim]
        return ring_conditions
    elif type(lower_lim) == float or type(lower_lim) == int:
        for i, j in zip(Li_hkl, Li_tth):
            ring_conditions[i] = [j, j + upper_lim, j - lower_lim]
        return ring_conditions
    else:
        raise ValueError(" lower_lim need to be either 'eq', int, or float!!!")

def get_im_array(im_path, flip=True):
    with Image.open(im_path) as im:
        if flip:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)  # flip it to match the GSAS tth,azm map
        return np.asarray(im)

def get_ring_array(hkl, im_array, ring_conditions,im_tth, im_azm, azm_range=None, detector=None):

    # im_array has been flipped to make it matches the GSAS-II image
    if detector == None:
        detector = "PE"

    ring_lst = ring_conditions[hkl]

    if azm_range is None:
        azm_min = 0
        azm_max = 360
    else:
        if len(azm_range) != 2:
            raise ValueError("The length of azm_range need to be 2, e.g. [10,30] ")
        azm_min, azm_max = azm_range

    r_max = ring_lst[1]
    r_min = ring_lst[2]
    ring_x = []
    ring_y = []
    ring_array = []
    ring_im = np.copy(im_array)

    if detector == 'PE':
        for x in range(2048):
            for y in range(2048):
                tth = im_tth[x, y]
                azm = im_azm[x, y]
                if r_min < tth < r_max and azm_min < azm < azm_max:
                    ring_x.append(2047 - x)
                    ring_y.append(y)
                    ring_array.append(im_array[x, y])
                    ring_im[x, y] = im_array[x, y]
                else:
                    ring_im[x, y] = 0
    elif detector == 'Pilatus':
        for x in range(1679):
            for y in range(1475):
                tth = im_tth[x, y]
                azm = im_azm[x, y]
                if r_min < tth < r_max and azm_min < azm < azm_max:
                    ring_x.append(1678 - x)
                    ring_y.append(y)
                    ring_array.append(im_array[x, y])
                    ring_im[x, y] = im_array[x, y]
                else:
                    ring_im[x, y] = 0
    ring_array = np.array(ring_array)

    return ring_x, ring_y, ring_array, ring_im

def get_ring_status(ring_array, remove_outlaw=False):

    # need to update the remove function
    if remove_outlaw:
        ring_array.sort()
        ring_array = ring_array[20:-20]

    # this is to remove the negative value for now
    # this can slow down the process and is not very logic
    # will update it in the future
    mean = ring_array.mean()
    ring_array = np.array([i for i in ring_array if i > mean * 0.5])  # ignore the negative values
    status = {'min': ring_array.min(),
              'max': ring_array.max(),
              'mean': ring_array.mean(),
              'std': ring_array.std()
              }
    return status
