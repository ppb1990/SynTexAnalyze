import os,sys
import numpy as np
import pandas as pd
from numba.core.typing.builtins import Range

from Python_files.current_version.multi_processing import out_dir_lst

from _utils import *

class TiffSetProcessor:
    def __init__(self,
                 tiff_dir, out_dir,
                 img_ctrl, img_ctrl_dir, im_tth=None, im_azm=None, get_tth_azm=True,
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

        # this is to handle the data from different beamlines
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

        # get the tth and azm calibration
        # either from loading calib files from GSAS-II
        # or from existing im_tth and im_azm files

        if detector is None:
            self.detector = 'PE'
        elif detector == 'Pilatus':
            self.detector = 'Pilatus'
        else:
            raise ValueError(f"The detector needs to be either PE or Pilatus for now!")

        print(self.tiff_dir)
        print(f"The detector is: {self.detector}")

        if get_tth_azm:
            print(f"Reading calibration from {img_ctrl}")
            self.GS_path = GS_path
            self.img_ctrl_path = os.path.join(img_ctrl_dir,img_ctrl)
            self.im_tth, self.im_azm = get_intTAmap(self.GS_path, self.file_path, self.img_ctrl_path)
        else:
            if im_tth is None:
                raise ValueError("im_tth is not found!")
            if im_azm is None:
                raise ValueError("im_azm is not found!")
            print("Calibration file loaded")
            self.im_tth = im_tth
            self.im_azm = im_azm

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
        self.ring_conditions = get_search_conditions(upper_lim, lower_lim, Li_lp_a, wavelength)
        if azm_range is not None:  # a list with two numbers, e.g. [90,270]
            self.azm_range = azm_range
        else:
            self.azm_range = None
        print(f"Initialization complete with wavelength: {self.wavelength}, search conditions: {upper_lim}")

    def get_mult_rings(self):
        # ongoing, obtain multiple rings from single process
        # this will save time and use the same im_array
        pass

    def get_ring(self, hkl, im_path, flip=True, detector=None, **kwargs):
        rc = self.ring_conditions
        im_tth = self.im_tth
        im_azm = self.im_azm
        azm_r = self.azm_range
        det = self.detector

        im_array = get_im_array(im_path, flip=flip)

        ring_x, ring_y, ring_array, ring_im = get_ring_array(hkl, im_array, rc, im_tth, im_azm, azm_r, det)

        if 'remove_outlaw' in kwargs.keys():
            ring_status = get_ring_status(ring_array, remove_outlaw=kwargs['remove_outlaw'])
        else:
            ring_status = get_ring_status(ring_array)

        return ring_x, ring_y, ring_array, ring_im, ring_status




