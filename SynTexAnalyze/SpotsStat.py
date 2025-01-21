import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from intensity_to_Ewald import *
import steo_projection


def get_processed_df(df, theta=2.1, operando=False, sphere='upper'):
    # Temporarily define a function here to process the raw spot data
    # Convert the azm and tilt angle to Phi and Chi
    # Will implement it to the tiff processing in the future update
    # Always assume inverse for now
    _, _, tth, azm, intensity, rotation = [df[i] for i in df.keys()]

    x, y, z = azmR_to_xyz(azm, rotation, R=1, theta=theta, operando=operando)
    c = {'tth': tth}
    x2, y2, z2, intensity2, dc = get_inverse(x, y, z, intensity, carrier=c)
    x3, y3, z3, intensity3, dc = get_sphere(x2, y2, z2, intensity2, sphere=sphere, carrier=dc)
    chi = steo_projection.to_SteoChi(z3)
    phi = steo_projection.to_SteoPhi(x3, y3)

    df = pd.DataFrame()
    keys = ['intensity', 'tth', 'Phi', 'Chi']
    values = [intensity3, dc['tth'], phi, chi]
    for key, value in zip(keys, values):
        df[key] = value

    return df


class SpotsStat:
    def __init__(self, path, name, out_dir, df_filter=None, auto_update=True):
        self.path = path
        self.name = name
        self._df = pd.read_csv(path)
        self._out_dir = out_dir
        self._filter = df_filter
        self._auto_update = auto_update
        if self._filter:
            title, sign, threshold = self._filter
            self._filtered_df = self.apply_filter(title, sign, threshold)
        else:
            self._filtered_df = None
        # need to add functions to calculate Phi and Chi here
        # using the initial tilting angle, azm, and the actual tilting angle

    @property
    def df(self):
        """Return the DataFrame."""
        return self._df

    @df.setter
    def df(self, new_df):
        if not isinstance(new_df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self._df = new_df

    @property
    def out_dir(self):
        """Return the output directory."""
        return self._out_dir

    @out_dir.setter
    def out_dir(self, new_out_dir):
        if not isinstance(new_out_dir, str):
            raise ValueError("out_dir must be a string")
        self._out_dir = new_out_dir

    @property
    def filter_df(self):
        """Return the filtered DataFrame based on the current filter."""
        if not self._filter:
            raise ValueError("Need a valid filter")
        return self._filtered_df

    @property
    def filter(self):
        """Return the filter."""
        if not self._filter:
            raise ValueError("No filter has ben set")
        return self._filter

    @filter.setter
    def filter(self, new_filter):

        if not isinstance(new_filter, list):
            raise ValueError("filter must be a list")
        elif len(new_filter) != 3:
            raise ValueError("filter must be the format of ['key', 'sign', value]")
        self._filter = new_filter
        if self._auto_update:
            title, sign, threshold = new_filter
            self._filtered_df = self.apply_filter(title, sign, threshold)

    def update(self):
        """
        This will update the filtered_df based on the current filter.

        :return: pandas.DataFrame: A filtered DataFrame based on the current filtering conditions.
        """
        if not self._filter:
            raise ValueError("Need a valid filter")
        key, sign, value = self._filter
        self._filtered_df = self.apply_filter(key, sign, value)

    def apply_filter(self, title, sign, threshold):
        """
        Apply a filter to the DataFrame based on the given key and sign

        Args:
            title (str): The column name in the DataFrame to apply the filter on. Default is 'intensity'
            sign (str): The comparing operator based on pandas.DataFrame operation.
            It should be one of '==', '!=', '<', '<=', '>=', '>'. Default is '>' which means larger than the threshold.
            threshold (int or float): the threshold value to filter the column against. Default is 100 for intensity.

        Returns:
            pandas.DataFrame: A filtered DataFrame based on the specified conditions.

        Raises:
            ValueError: If the specified comparing operator is invalid
        """

        def le(name, value):
            return self.df[self.df[name].le(value)]

        def lt(name, value):
            return self.df[self.df[name].lt(value)]

        def ge(name, value):
            return self.df[self.df[name].ge(value)]

        def gt(name, value):
            return self.df[self.df[name].gt(value)]

        def eq(name, value):
            return self.df[self.df[name].eq(value)]

        def ne(name, value):
            return self.df[self.df[name].ne(value)]

        dispatcher = {'==': eq, '!=': ne, '<=': le, '<': lt, '>=': ge, '>': gt}

        try:
            return dispatcher[sign](title, threshold)
        except KeyError:
            raise ValueError(f"Invalid sign: {sign}. Valid signs are {list(dispatcher.keys())}.")

    def get_statistic(self, title, filtering=False):
        if filtering:
            value = self._filtered_df[title]
        else:
            value = self._df[title]
        count, mean, std, _min, _, _, _, _max = value.describe()
        dc = {'count': count,
              'mean': mean,
              'std': std,
              'min': _min,
              'max': _max
              }
        return dc

    def get_inten_hist(self, bins='default', log=False, get_st=True, filtering=False):  # get statistic of intensity
        """Get the intensity distribution based on pre-defined bins"""
        if bins == 'default':
            # bins = [10,100,1000,10000,100000,1000000]
            bins = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
            distribution = [0] * len(bins)
            if filtering:
                intensity = self._filtered_df['intensity']
            else:
                intensity = self._df['intensity']
            for i in intensity:
                a = int(np.log10(i))
                idx = (a * 2 - 1) + int(np.log10(i / 10 ** a) - np.log10(5) + 1)
                if idx >= 10:
                    idx = 10
                distribution[idx] += 1
            if log:
                distribution = [np.log10(i + 1) for i in distribution]
            if get_st:
                st = self.get_statistic('intensity')
                return distribution, bins, st
            else:
                return distribution, bins
        else:
            pass  # working on it

    def plt_inten_hist(self, bins='default', log=False, range=None, show=True,
                       save_as=None, get_st=True, filtering=False,
                       **kwargs):
        """
        Plot the intensity histogram of the spots


        Args:
            bins (str)
            log (bool)
            range (None or list)
            show (bool)
            save_as (str)
            get_st (bool)
            filtering (bool)

        Returns:
            pandas.DataFrame (optional)

        """
        if get_st:
            distribution, _, st = self.get_inten_hist(bins=bins, log=log, get_st=get_st, filtering=filtering)
        else:
            distribution, _ = self.get_inten_hist(bins=bins, log=log, get_st=get_st, filtering=filtering)

        fig, ax = plt.subplots()
        if not range:
            range = ['<10', '50', '100', '500', '1k', '5k', '10k', '50k', '100k', '500k', '1m']
        ax.bar(range, distribution)

        if 'xlabel' not in kwargs.keys():
            ax.set_xlabel('Intensity')
        else:
            ax.set_xlabel(kwargs['xlabel'])

        if 'ylabel' not in kwargs.keys():
            ax.set_ylabel('Counts')
        else:
            ax.set_ylabel(kwargs['ylabel'])

        if 'ylim' not in kwargs.keys():
            pass
        else:
            if not isinstance(kwargs['ylim'], list):
                raise ValueError('ylim needs to be a list')
            if len(kwargs['ylim'] != 2):
                raise ValueError('the length of ylim needs to be 2 for min and max values')
            ymin, ymax = kwargs['ylim']
            ax.set_ylim(ymin=ymin, ymax=ymax)

        if 'title' not in kwargs.keys():
            # ax.set_title('Intensity')
            pass
        else:
            ax.set_title(kwargs['title'])

        if save_as:
            plt.savefig(self.out_dir + save_as)

        if show:
            plt.show()
        else:
            plt.close(fig)
        if get_st:
            return st

    def get_tth_hist(self, bin_size=20, filtering=False, get_st=False):
        """
        Get the tth histogram
        """
        if filtering:
            tth = self._filtered_df['tth']
        else:
            tth = self._df['tth']
        distribution, bins = np.histogram(tth, bins=bin_size)
        bins = bins[:-1]  # remove the rightmost edge
        if get_st:
            st = self.get_statistic(title='tth', filtering=filtering)
            return distribution, bins, st
        else:
            return distribution, bins

    def plt_tth_hist(self, bin_size=20, show=True, save_as=None, get_st=True, filtering=False, **kwargs):
        """
        Plot the intensity histogram of the spots


        Args:
            bin_size (int)
            show (bool)
            save_as (str)
            get_st (bool)
            filtering (bool)

        Returns:
            pandas.DataFrame (optional)

        """
        if get_st:
            distribution, bins, st = self.get_tth_hist(bin_size=bin_size, filtering=filtering, get_st=get_st)
        else:
            distribution, bins = self.get_tth_hist(bin_size=bin_size, filtering=filtering, get_st=get_st)
        bins = [str(format(i, '.3f')) for i in bins]  # change bins to str

        fig, ax = plt.subplots()
        ax.bar(bins, distribution)
        plt.xticks(rotation=-45)

        if 'xlabel' not in kwargs.keys():
            ax.set_xlabel('tth')
        else:
            ax.set_xlabel(kwargs['xlabel'])

        if 'ylabel' not in kwargs.keys():
            ax.set_ylabel('Counts')
        else:
            ax.set_ylabel(kwargs['ylabel'])

        if 'title' not in kwargs.keys():
            # ax.set_title('Intensity')
            pass
        else:
            ax.set_title(kwargs['title'])

        if save_as:
            plt.savefig(self.out_dir + save_as)

        if show:
            plt.show()
        else:
            plt.close(fig)
        if get_st:
            return st

    def get_phi_hist(self, bin_size=30, filtering=False, get_st=False):
        """
        Get the phi histogram
        phi ranging from 0-360
        """
        if filtering:
            phi = self._filtered_df['Phi']
        else:
            phi = self._df['Phi']
        distribution, bins = np.histogram(phi, bins=bin_size)
        bins = bins[:-1]  # remove the rightmost edge
        if get_st:
            st = self.get_statistic(title='Phi', filtering=filtering)
            return distribution, bins, st
        else:
            return distribution, bins

    def plt_phi_hist(self, bin_size=30, show=True, save_as=None, get_st=True, filtering=False, **kwargs
                     ):
        """
        Plot the intensity histogram of the spots


        Args:
            bin_size (int)
            show (bool)
            save_as (str)
            get_st (bool)s
            filtering (bool)

        Returns:
            pandas.DataFrame (optional)

        """
        if get_st:
            distribution, bins, st = self.get_phi_hist(bin_size=bin_size, filtering=filtering, get_st=get_st)
        else:
            distribution, bins = self.get_phi_hist(bin_size=bin_size, filtering=filtering, get_st=get_st)
        bins = [str(format(i, '.3f')) for i in bins]  # change bins to str

        fig, ax = plt.subplots()
        ax.bar(bins, distribution)
        plt.xticks(rotation=-45)

        if 'xlabel' not in kwargs.keys():
            ax.set_xlabel('Phi')
        else:
            ax.set_xlabel(kwargs['xlabel'])

        if 'ylabel' not in kwargs.keys():
            ax.set_ylabel('Counts')
        else:
            ax.set_ylabel(kwargs['ylabel'])

        if 'title' not in kwargs.keys():
            # ax.set_title('Intensity')
            pass
        else:
            ax.set_title(kwargs['title'])

        if save_as:
            plt.savefig(self.out_dir + save_as)

        if show:
            plt.show()
        else:
            plt.close(fig)
        if get_st:
            return st

    def get_chi_hist(self, bin_size=30, filtering=False, get_st=False):
        """
        Get the chi histogram
        chi ranging from 0-90
        """
        if filtering:
            chi = self._filtered_df['Chi']
        else:
            chi = self._df['Chi']
        distribution, bins = np.histogram(chi, bins=bin_size)
        bins = bins[:-1]  # remove the rightmost edge
        if get_st:
            st = self.get_statistic(title='Chi', filtering=filtering)
            return distribution, bins, st
        else:
            return distribution, bins

    def plt_chi_hist(self, bin_size=30, show=True, save_as=None, get_st=True, filtering=False, **kwargs):
        """
        Plot the intensity histogram of the spots


        Args:
            bin_size(int)
            show (bool)
            save_as (str)
            get_st (bool)
            filtering (bool)

        Returns:
            pandas.DataFrame (optional)

        """
        if get_st:
            distribution, bins, st = self.get_chi_hist(bin_size=bin_size, filtering=filtering, get_st=get_st)
        else:
            distribution, bins = self.get_chi_hist(bin_size=bin_size, filtering=filtering, get_st=get_st)
        # change bins to str
        bins = [str(format(i, '.0f')) for i in bins]

        fig, ax = plt.subplots()
        ax.bar(bins, distribution)
        plt.xticks(rotation=-45)

        if 'xlabel' not in kwargs.keys():
            ax.set_xlabel('Phi')
        else:
            ax.set_xlabel(kwargs['xlabel'])

        if 'ylabel' not in kwargs.keys():
            ax.set_ylabel('Counts')
        else:
            ax.set_ylabel(kwargs['ylabel'])

        if 'title' not in kwargs.keys():
            # ax.set_title('Intensity')
            pass
        else:
            ax.set_title(kwargs['title'])

        if save_as:
            plt.savefig(self.out_dir + save_as)

        if show:
            plt.show()
        else:
            plt.close(fig)
        if get_st:
            return st

