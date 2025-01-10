import os
import numpy as np
import pandas as pd
import math
import sys
import shutil
from PIL import Image
from matplotlib import pyplot as plt
import time

from pandas.core.config_init import pc_show_dimensions_doc


class synchrtron_imgSet_processing:  # making a class for this function
    def __init__(self,
                 tiff_dir, out_dir, img_ctrl, img_ctrl_dir,
                 system='windows', GS_path='default',
                 upper_lim=0.015, lower_lim='eq',
                 wavelength=0.1819, Li_lp_a=3.507,
                 file_type='texture', azm_range=None
                 ):
        # file path for single data set
        if system == 'windows':
            if tiff_dir[-1] == '\\':
                self.tiff_dir = tiff_dir + 'dark_sub\\'
            else:
                self.tiff_dir = tiff_dir + '\\dark_sub\\'
        elif system == 'mac':
            if tiff_dir[-1] == '/':
                self.tiff_dir = tiff_dir + 'dark_sub/'
            else:
                self.tiff_dir = tiff_dir + '/dark_sub/'
        else:
            raise ValueError("'system' needs to be either 'windows' or 'mac'!!!")

        if GS_path == 'default':
            self.GS_path = 'C:\XRD\GSAS-II\GSAS-II\GSASII'
        elif type(GS_path) == str:
            self.GS_path = GS_path
        else:
            raise ValueError(" 'GS_path' needs to be a dir!!!")

        self.file_lst = [i for i in os.listdir(self.tiff_dir) if '.tiff' in i]
        self.file_path = [self.tiff_dir + i for i in self.file_lst]
        self.out_dir = out_dir
        self.img_ctrl_path = os.path.join(img_ctrl_dir, img_ctrl)
        self.file_type = file_type
        if file_type == 'texture':
            self.SetName = self.tiff_dir.split('texture_')[1].split('_f')[0]
        elif file_type == 'Texture':  # temp fix for the ONR data
            self.SetName = self.tiff_dir.split('Texture_')[1]
        elif file_type == 'mapping':
            self.SetName = self.tiff_dir.split('AF_')[1].split('_mapping')[0]
        elif file_type == 'Operando':
            self.SetName = self.tiff_dir.split('\\')[-3]

        self.wavelength = wavelength
        self.Li_lp_a = Li_lp_a
        self.im_tth, self.im_azm = self.ZL_get_intTAmap()
        self.ring_conditions = self.get_search_conditions(upper_lim, lower_lim, Li_lp_a, wavelength)
        if azm_range != None:  # a list with two numbers, e.g. [90,270]
            self.azm_range = azm_range
        else:
            self.azm_range = None
        print(f"Initialization complete with wavelength: {self.wavelength}, search conditions: {upper_lim}")

    def d_to_tth(self, d, w):  # calculate the tth from d spacing
        tth = np.arcsin(w / (d * 2)) * 2  # results in radians
        tth = tth * 180 / np.pi
        return tth

    def get_rotation_th(self, file_name):  # for texture sample, get rotation angle
        rotation_th = file_name.split('th_')[1].split(',')[0]
        return rotation_th

    def get_psn_xy(self, file_name):  # for Austin's sample, get psn_x, psn_y
        psn_x = file_name.split('sample_x_')[1].split('mm')[0].replace(',', '.')
        psn_y = file_name.split('sample_y_')[1].split('mm')[0].replace(',', '.')
        psn_x = float(psn_x)
        psn_y = float(psn_y)
        return psn_x, psn_y

    def get_tth_azm(self, intTAmap):
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

        return im_tth, im_azm

    def ZL_get_intTAmap(self):

        sys.path.insert(0, self.GS_path)  # needed to "find" GSAS-II modules
        import GSASIIscriptable as G2sc
        import gc

        cache = {}
        file_path = self.file_path[0]  # temp use
        # ctrl_path = os.path.join('D://Postdoc//XPD-2024-2//Zhuo','texture-control.imctrl')

        gpx = G2sc.G2Project(newgpx='temp.gpx')  # temp use
        img = gpx.add_image(file_path, cacheImage=False)  # don't need to cache image
        img[0].loadControls(self.img_ctrl_path)

        # cache['intMaskMap'] = img[0].IntMaskMap()
        cache['intTAmap'] = img[0].IntThetaAzMap()

        im_tth, im_azm = self.get_tth_azm(cache['intTAmap'])

        del cache, gpx

        return im_tth, im_azm

    def get_search_conditions(self,
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
        Li_tth = [self.d_to_tth(i, wavelength) for i in Li_d]

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

    def ZL_get_img_array(self, im_path,flip=True):
        if flip != True:
            im_array = np.array(Image.open(im_path))
        else:
            im_flip = Image.open(im_path).transpose(Image.FLIP_TOP_BOTTOM)  # flip it to match the GSAS tth,azm map
            im_array = np.array(im_flip)
            del im_flip
        return im_array

    def ZL_get_ring_status(self,ring_array, remove_outlaw=False): #  get the statistics of the ring region
        if remove_outlaw:
            ring_array.sort()
            ring_array = ring_array[20:-20]  # remove 20 smallest and 20 biggest number
        mean = ring_array.mean()
        ring_array = np.array([i for i in ring_array if i > mean * 0.5])  # ignore the negative values
        status = {'min':ring_array.min(),
                 'max' :ring_array.max(),
                 'mean':ring_array.mean(),
                 'std' :ring_array.std()
                 }
        return status

    def ZL_get_ring(self, hkl, im_array):
        # im_array has been flipped to make it matches the GSAS-II image
        # check to see if tth = im_tth[x,y] will return the correct image

        # Aug-19 2024 added the filter option for azm range

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
        ring_array = np.array(ring_array)
                    
        ring_status = self.ZL_get_ring_status(ring_array)

        return ring_x, ring_y, ring_array, ring_im, ring_status
    def ZL_ring_histogram(self,ring_array):
        hist = {}
        for i in ring_array:
            if not i in hist.keys():
                hist[i] = 1
            else:
                hist[i] +=1
        return hist
        
    def ZL_ring_hist_fig(self, hist, xlim=None, ylim=None, title=None, img_show=False, img_path=False):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # value = [np.log10(i*10) for i in hist.values()]
        # ax.bar(hist.keys(), value)
        ax.bar(hist.keys(),hist.values())
        ax.set_xlabel('intensity')
        ax.set_ylabel('counts')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title)
        if img_show:
            plt.show()
        
        if img_path:
            fig.savefig(img_path + '.jpg')
            plt.close('all')
    

    def ZL_spot_analysis(self, im_processed_array):
        # import the package for processing
        from skimage import measure
        
        binary_image = im_processed_array > 0
        labels = measure.label(binary_image, connectivity=2)
        spot_properties = measure.regionprops(labels, intensity_image=im_processed_array)
        del binary_image, labels
        return spot_properties
        
    def ZL_get_spot_xy(self, ring_array, ring_x, ring_y, im_array,
                       filter_method=None,
                       double=False,
                       **kwargs):

        if filter_method is None:
            filter_method = {'mean': 2.75}
        if 'out_dir' not in kwargs.keys():
            out_dir = self.out_dir
        else:
            out_dir = kwargs['out_dir']
        
        # define the filter 
        if 'std' in filter_method.keys():
            status = self.ZL_get_ring_status(ring_array)
            mean = status['mean']
            std = status['std']
            ratio = filter_method['std']
            f = mean + std * ratio
        elif 'mean' in filter_method.keys():
            status = self.ZL_get_ring_status(ring_array)
            mean = status['mean']
            ratio = filter_method['mean']
            f = mean * ratio
        elif 'hist' in filter_method.keys():
            # find the most counted intensity through histogram
            hist = filter_method['hist']
            ratio = filter_method['ratio']
            
            count = [i for i in hist.values()]
            #  find the most counted intensity
            count_max = max(count)  
            #  find the relative intensity (peak intensity)
            mean = [i for i in hist if hist[i] == count_max][0]
            # print(max_intensity)
            f = mean * ratio   

        # get spot x,y
        print('BGK intensity: {}, filter intensity: {}'.format(mean, f))
        spot_x = []
        spot_y = []
        
        for x,y,i in zip(ring_x, ring_y, ring_array):
            if i >f:
                spot_x.append(2047-x)
                spot_y.append(y)
        #####
        # add a function to do double filter
        if double:
            for ix, iy in zip(spot_x,spot_y):
                pass

            pass
        # get the processed image
        im_processed_array = im_array * 0
        for x,y in zip(spot_x,spot_y):
            im_processed_array[x,y] = im_array[x,y]
            
        spot_properties = self.ZL_spot_analysis(im_processed_array)

        # applying another searching here based on spot image
        # 5 x 5 or 10 x 10
        # the signal should be above the average
        
        return spot_x, spot_y, im_processed_array, spot_properties, mean

    def ZL_get_spots_status(self, spot_properties, mean, im_array,info,
                            get_spot_img=True,
                            img_show=False, save=True,
                            **kwargs
                           ):  # ongoing
        from skimage import measure, filters, morphology

        if 'out_dir' not in kwargs.keys():
            out_dir = self.out_dir
        else:
            out_dir = kwargs['out_dir']

        out_dc = {
            'count':[],
            'rotation':[],
            'tth':[],
            'azm':[],
            'intensity':[]
                 }
                    
        if self.file_type == 'texture':
            out_dc['rotation'] = []
            f_name, rotation_th = [i for i in info]
            
        if self.file_type == 'Texture':
            out_dc['rotation'] = []
            f_name, rotation_th = [i for i in info]
            
        if self.file_type == 'mapping':  # add the psn_x, psn_y for each data point
            out_dc['mapping_x'] = []
            out_dc['mapping_y'] = []
            f_name, mapping_x, mapping_y = [i for i in info]
        if self.file_type == 'Operando':
            out_dc['rotation'] = []
            f_name,rotation_th = [i for i in info]
        
        total_counts = len(spot_properties)
        # mean is the background, used the same intensity as the Zl_get_spot_xy methode
        count = 0
        
        if total_counts >0:
            for p in spot_properties:
                x,y = [int(k) for k in p.centroid]
                intensity = 0

                for k in p.coords:  # calculate the total intensity, no averaging
                    xk,yk = k
                    intensity += im_array[xk,yk]
                    intensity -= mean
                out_dc['count'].append(count)
                out_dc['tth'].append(self.im_tth[x,y])
                out_dc['azm'].append(self.im_azm[x,y])
                out_dc['intensity'].append(intensity)
                if self.file_type == 'texture':
                    out_dc['rotation'].append(rotation_th)  # self.get_rotation_th(file_name)

                if self.file_type == 'mapping':  # add the psn_x, psn_y for each data point
                    out_dc['mapping_x'].append(mapping_x)
                    out_dc['mapping_y'].append(mapping_y)

                if self.file_type == 'Operando':
                    out_dc['rotation'].append(rotation_th)


                count +=1
        ## get spot imag for each data set if get_spot_img == True
        if get_spot_img:
            spot_img_name = f_name
            self.ZL_get_spot_img(im_array, spot_properties, spot_img_name, out_dc['intensity'], img_show=img_show, save=save,
                                 out_dir=out_dir
                                )
        
        return out_dc,total_counts


    def ZL_get_spot_img(self, im_array, spot_properties, img_name, intensity_lst, img_show = False, save = True, **kwargs):
        # this should be put within the get spot properties function
        if 'out_dir' not in kwargs.keys():
            out_dir = self.out_dir
        else:
            out_dir = kwargs['out_dir']
        
            
        img_len = len(spot_properties)//3 +1

        # set the img_len to 9 if its larger than 9
        if img_len >=9: 
            img_len = 9
        
        fig,axs = plt.subplots(img_len, 3, figsize=(10, img_len*3.33), constrained_layout=True)
        idx = 0
        if len(spot_properties) != 0:
            for i in spot_properties:
                if idx//3+1 <= img_len:
                    x,y = [int(j) for j in i.centroid]
                    spot = im_array[x-5:x+6,y-5:y+6]
                    intensity = intensity_lst[idx]
                    if img_len !=1:
                        ax_x = idx//3
                        ax_y = idx%3
                        axs[ax_x,ax_y].imshow(spot,aspect = 'auto')
                        axs[ax_x,ax_y].title.set_text('# {}\nintensity{}\ntth_{:.3f}\nazm_{:.2f}'.format(str(idx),str(intensity),float(self.im_tth[x,y]),float(self.im_azm[x,y])))
                    elif img_len == 1:
                        ax_x = idx%3
                        axs[ax_x].imshow(spot,aspect = 'auto')
                        axs[ax_x].title.set_text('# {}\nintensity{}\ntth_{:.3f}\nazm_{:.2f}'.format(str(idx),str(intensity),float(self.im_tth[x,y]),float(self.im_azm[x,y])))
                
                    idx +=1
            
                    fig.suptitle(img_name + '_spot')
            
        if save == True:
            fig.savefig(out_dir + img_name + '.jpg')
            
        if img_show == True:
            fig.show()
        
        #plt.close('all')

    def ZL_img_processing(self,path, 
                          flip=False, hkl='110',
                          get_hist=True, hist_show=False,
                          f_method='hist', ratio=2.5,
                          get_spot_img=True, spot_show=False,
                          **kwargs
                         ):
        if self.file_type == 'texture':
            rotation_th = self.get_rotation_th(path)
            f_name = self.SetName + '_' + rotation_th + '_' + hkl
            info = [f_name,rotation_th]
        if self.file_type == 'Texture':
            rotation_th = self.get_rotation_th(path)
            f_name = self.SetName + '_' + rotation_th + '_' + hkl
            info = [f_name,rotation_th]
        elif self.file_type == 'mapping':
            mapping_x,mapping_y = self.get_psn_xy(path)
            f_name = self.SetName + '_x_{}_y_{}_{}'.format(mapping_x,mapping_y,hkl)
            info = [f_name,mapping_x,mapping_y]
        elif self.file_type == 'Operando':
            rotation_th = self.get_rotation_th(path)            
            f_name = self.SetName + '_' +  rotation_th + '_' + hkl            
            info = [f_name,rotation_th]

        if 'out_dir' not in kwargs.keys():
            out_dir = self.out_dir
        else:
            out_dir = kwargs['out_dir']
            
        print('\nProcessing: {}'.format(f_name))
        # out_dir = self.out_dir
        im_array = self.ZL_get_img_array(path,flip=flip)
        ring_x,ring_y,ring_array,ring_im,ring_status = self.ZL_get_ring(hkl,im_array)
        ring_hist = self.ZL_ring_histogram(ring_array)
        # generate the f_method dc based on the filter method and the ratio
        if f_method == 'hist':
            f_method = {'hist': ring_hist, 'ratio': ratio}
        else:
            f_method = {f_method: ratio}
        if get_hist:
            self.ZL_ring_hist_fig(ring_hist,img_path=out_dir + f_name + '_ring_hist',
                                  title=f_name + '_ring_hist', img_show=hist_show)
        spot_x,spot_y,processed_array,spot_p,mean_intensity = self.ZL_get_spot_xy(ring_array, ring_x, ring_y, im_array,
                                                                                  filter_method=f_method)
        out_dc, total_counts = self.ZL_get_spots_status(spot_p, mean_intensity, im_array, info,
                                                        get_spot_img=get_spot_img, img_show=spot_show, out_dir=out_dir)
        return f_name, out_dc, total_counts

    def ZL_imgSet_processing(self,
                             flip=False, hkl='110',
                             get_hist=True, hist_show=False,
                             f_method='hist', ratio=2.5,
                             get_spot_img=True, spot_show=False,
                             iteration=None,
                             **kwargs
                             ):
        # 8/24, write the code to repeat spot searching if a certain condition is not meet
        # default is to process Li 110

                
        #im_tth = self.im_tth
        #im_azm = self.im_azm
        
        if 'file_lst' not in kwargs.keys():
            file_lst = self.file_lst
        else:
            file_lst = kwargs['file_lst']

        if 'file_path' not in kwargs.keys():
            file_path = self.file_path
        else:
            file_path = kwargs['file_path']

        if 'out_dir' not in kwargs.keys():
            out_dir = self.out_dir
        else:
            out_dir = kwargs['out_dir']

        # print(file_lst,file_path)
        
        
        over_all_time = time.time()
        spot_df = pd.DataFrame()
        spot_dc = {
            'count':[],
            'tth':[],
            'azm':[],
            'intensity':[]
                  }
        if self.file_type == 'texture':
            spot_dc['rotation'] = []
        
        if self.file_type == 'mapping':  # add the psn_x, psn_y for each data point
            spot_dc['mapping_x'] = []
            spot_dc['mapping_y'] = []
            
        # img_idx = 0  # number of image
        total_spots = 0  # total spots 
        
        for file, path in zip(file_lst, file_path):
            t0 = time.time()
            f_name, out_dc, spot_counts = self.ZL_img_processing(path,
                                                                 flip=flip, hkl=hkl,
                                                                 get_hist=get_hist, hist_show=hist_show,
                                                                 f_method=f_method, ratio=ratio,
                                                                 get_spot_img=get_spot_img, spot_show=spot_show,
                                                                 out_dir=out_dir
                                                                 )

            if type(iteration) == list and len(iteration) ==3 and spot_counts > iteration[2] :
                # a list like parameter contains one int, and one float
                # the first one is the # of iteration and the sign of iteration
                # the second one is the change in searching condition during iteration (always positive)
                # r = r0 * (1 + n * x)   (abs(x) to make sure it is always positive
                idx = 0
                loop_n = abs(iteration[0])
                sign = iteration[0]/abs(iteration[0])
                change = abs(iteration[1])
                cutoff = iteration[2]  # for now, set up the cut off manually

                while idx < loop_n:
                    ratio_new = ratio * (1 + sign * (idx+1) * change)
                    f_name, out_dc, spot_counts = self.ZL_img_processing(path,
                                                                         flip=flip, hkl=hkl,
                                                                         get_hist=get_hist, hist_show=hist_show,
                                                                         f_method=f_method, ratio=ratio_new,
                                                                         get_spot_img=get_spot_img, spot_show=spot_show,
                                                                         out_dir=out_dir
                                                                         )
                    if spot_counts <= cutoff:
                        break
                    else:
                        idx +=1

            for key in spot_dc:
                if key == 'count':
                    spot_dc[key] = spot_dc[key] + [i+total_spots for i in out_dc[key]]
                else: 
                    spot_dc[key] = spot_dc[key] + [i for i in out_dc[key]]
            # img_idx +=1
            total_spots += spot_counts
            plt.close('all')
            print('{}Finished, process time: {:.2f}\n'.format(f_name,time.time()-t0))
                  
        print('Image processing finished, total spots found: {}, total time: {:.2f}s.'.format(total_spots,time.time()-over_all_time))
        for key in spot_dc:
            if key != 'total_counts':
                spot_df[key] = spot_dc[key]
        spot_df.to_csv(out_dir + f_name + '_total_{}.csv'.format(total_spots))


        #return spot_df 
        
        

    

