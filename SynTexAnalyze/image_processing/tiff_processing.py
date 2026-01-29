import os,sys
import numpy as np
import pandas as pd
from numba.core.typing.builtins import Range

from Python_files.current_version.multi_processing import out_dir_lst

from utils import *

class TiffSetProcessor:
    def __init__(self,
                 tiff_dir, out_dir, img_ctrl, img_ctrl_dir,
                 root='C:\\', GS_path='default',
                 upper_lim=0.015, lower_lim='eq',
                 wavelength=0.1819,
                 Li_lp_a=3.507,
                 file_type='texture', azm_range=None,
                 no_dark_sub=False, SetName=None,
                 detector=None
                 ):
        # file path for single data set

        # need to modify the "Li_li_p" to another parameter
        # so it can be used for other structure

        tiff_dir = tiff_dir.rstrip("\\/")  # clean the path

        if not no_dark_sub:
            tiff_dir = os.path.join(tiff_dir,"dark_sub")
        self.tiff_dir = tiff_dir

        self.file_lst = [f for f in os.listdir(tiff_dir) if f.lower().endswith((".tif", ".tiff"))]
        self.file_path = [os.path.join(tiff_dir,f) for f in self.file_lst]
        self.out_dir = out_dir

        if GS_path == 'default':
            if sys.platform.startswith("win"):  # for Windows system
                GS_path = r'C:\XRD\GSAS-II\GSAS-II\GSASII'
                if not os.path.isdir(GS_path):
                    raise FileNotFoundError(f"GSAS-II path not found: {GS_path}")
            else:
                raise ValueError("Please specify the GSAS-II path")
        elif GS_path == 'auto':
            def find_gsasii(root):
                for dir_path, dir_names, _ in os.walk(root):
                    if "GSAS-II" in dir_names:
                        return os.path.join(dir_path, "GSAS-II")
                raise FileNotFoundError(f"GSAS-II was not detected, please specify your GSAS-II path!!!")
            GS_path = find_gsasii(root)
        else:
            if not os.path.isdir(GS_path):
                raise FileNotFoundError(f"GSAS-II path not found: {GS_path}")

        self.GS_path = GS_path
        self.img_ctrl_path = os.path.join(img_ctrl_dir,img_ctrl)

        if detector is None:
            self.detector = 'PE'
        elif detector == 'Pilatus':
            self.detector = 'Pilatus'
        else:
            raise ValueError(f"The detector needs to be either PE or Pilatus for now!")

        print(self.tiff_dir)
        print('The detector is: {}.'.format(self.detector))
        print(f"The image control file is: {img_ctrl}")

        # temp fix for the file type for now, in the future will make it more universal.
        # should make a function to handle the data set more properly
        self.file_type = file_type
        if file_type == 'texture':
            if SetName:
                self.SetName = SetName
            else:
                self.SetName = self.tiff_dir.split('texture_')[1].split('_f')[0]
        elif file_type == 'Texture':  # temp fix for the ONR data
            self.SetName = self.tiff_dir.split('Texture_')[1].split('_f')[0]
        elif file_type == 'mapping':
            self.SetName = self.tiff_dir.split('AF_')[1].split('_mapping')[0]
        elif file_type == 'Operando':
            self.SetName = self.tiff_dir.split('\\')[-3]

        self.wavelength = wavelength  # define the wavelength in angstroms
        self.Li_lp_a = Li_lp_a  # default is lithium and the lithium
        self.im_tth, self.im_azm = get_intTAmap(self.GS_path, self.file_path, self.img_ctrl_path)
        self.ring_conditions = get_search_conditions(upper_lim, lower_lim, Li_lp_a, wavelength)
        if azm_range is not None:  # a list with two numbers, e.g. [90,270]
            self.azm_range = azm_range
        else:
            self.azm_range = None
        print(f"Initialization complete with wavelength: {self.wavelength}, search conditions: {upper_lim}")

    def ZL_get_ring(self, hkl, im_array, detector=None):
        # im_array has been flipped to make it matches the GSAS-II image
        # check to see if tth = im_tth[x,y] will return the correct image

        # Aug-19 2024 added the filter option for azm range
        if detector == None:
            detector = self.detector

        ring_lst = self.ring_conditions[hkl]
        if self.azm_range != None:
            azm_min = self.azm_range[0]
            azm_max = self.azm_range[1]
        else:
            # no limitation
            azm_min = 0
            azm_max = 360

        r_max = ring_lst[1]  # [j,j+upper_lim,j-upper_lim]
        r_min = ring_lst[2]
        ring_x = []
        ring_y = []
        ring_array = []
        ring_im = np.copy(im_array)

        if detector == 'PE':
            for x in range(2048):
                for y in range(2048):
                    tth = self.im_tth[x, y]
                    azm = self.im_azm[x, y]
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
                    tth = self.im_tth[x, y]
                    azm = self.im_azm[x, y]
                    if r_min < tth < r_max and azm_min < azm < azm_max:
                        ring_x.append(1678 - x)
                        ring_y.append(y)
                        ring_array.append(im_array[x, y])
                        ring_im[x, y] = im_array[x, y]
                    else:
                        ring_im[x, y] = 0
        ring_array = np.array(ring_array)

        ring_status = self.ZL_get_ring_status(ring_array)

        return ring_x, ring_y, ring_array, ring_im, ring_status


