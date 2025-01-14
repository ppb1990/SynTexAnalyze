import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class SpotsStat:
    def __init__(self, path, name, out_dir, df_filter=None, auto_update=True):
        self.path = path
        self.name = name
        self._df = pd.read_csv(path)
        self._out_dir = out_dir
        self._filter = df_filter
        self._auto_update = auto_update
        if self._filter:
            key, sign, value = self._filter
            self._filtered_df = self.apply_filter(key, sign, value)
        else:
            self._filtered_df = None

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
            key, sign, value = new_filter
            self._filtered_df = self.apply_filter(key, sign, value)

    def update(self):
        """
        This will update the filtered_df based on the current filter.

        :return: pandas.DataFrame: A filtered DataFrame based on the current filtering conditions.
        """
        if not self._filter:
            raise ValueError("Need a valid filter")
        key, sign, value = self._filter
        self._filtered_df = self.apply_filter(key, sign, value)

    def apply_filter(self, key='intensity', sign='<', value=100):
        """
        Apply a filter to the DataFrame based on the given key and sign

        Args:
            key (str): The column name in the DataFrame to apply the filter on. Default is 'intensity'
            sign (str): The comparision operator based on pandas.DataFrame operation, it should be one of '==', '!=', '<', '<=', '>=', '>'. Default is '<'.
            value (int or float): the threshold value to filter the column against. Default is 100 for intensity.

        Returns:
            pandas.DataFrame: A filtered DataFrame based on the specified conditions.

        Raises:
            ValueError: If the specified comparision operator is invalid
        """

        def le(key, value):
            return self.df[self.df[key].le(value)]

        def lt(key, value):
            return self.df[self.df[key].lt(value)]

        def ge(key, value):
            return self.df[self.df[key].ge(value)]

        def gt(key, value):
            return self.df[self.df[key].gt(value)]

        def eq(key, value):
            return self.df[self.df[key].eq(value)]

        def ne(key, value):
            return self.df[self.df[key].ne(value)]

        dispatcher = {'==': eq, '!=': ne, '<=': le, '<': lt, '>=': ge, '>': gt}

        try:
            return dispatcher[sign](key, value)
        except KeyError:
            raise ValueError(f"Invalid sign: {sign}. Valid signs are {list(dispatcher.keys())}.")

    def get_statistic(self, key):
        value = self._df[key]
        count, mean, std, _min, _, _, _, _max = value.describe()
        dc = {'count': count,
              'mean': mean,
              'std': std,
              'min': _min,
              'max': _max
              }
        return dc

    def gs_intensity(self, bins='default', log=False):  # get statistic of intensity
        if bins == 'default':
            # bins = [10,100,1000,10000,100000,1000000]
            bins = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
            st = [0 for i in bins]
            intensity = self._df['intensity']
            for i in intensity:
                E = int(np.log10(i))
                idx = (E * 2 - 1) + int(np.log10(i / 10 ** E) - np.log10(5) + 1)
                if idx >= 10:
                    idx = 10
                st[idx] += 1
            if log:
                st = [np.log10(i + 1) for i in st]
            return st, bins
        else:
            pass  # working on it

    def plt_inten_hist(self, bins='default', log=False, range=None, show=True, saveAs=None,
                       **kwargs):  # plot intensity histogram
        st, _ = self.gs_intensity(bins=bins, log=log)  # not using the actually number here for x range
        fig, ax = plt.subplots()
        if not range:
            range = ['<10', '50', '100', '500', '1k', '5k', '10k', '50k', '100k', '500k', '1m']
        ax.bar(range, st)

        if 'xlabel' not in kwargs.key():
            ax.set_xlabel('Intensity')
        else:
            ax.set_xlabel(kwargs['xlabel'])
        if 'ylabel' not in kwargs.key():
            ax.set_ylabel('Intensity')
        else:
            ax.set_ylabel(kwargs['xlabel'])
        if 'title' not in kwargs.key():
            # ax.set_title('Intensity')
            pass
        else:
            ax.set_title(kwargs['title'])

        if show:
            plt.show()

        if saveAs:
            plt.savefig(self.out_dir + saveAs)

    def gs_tth(self, bins=20):
        pass

    def plt_tth_hist(self):
        pass

    def gs_phi(self, bin=30):
        pass

    def plt_phi_hist(self):
        pass

    def gs_chi(self, bin=30):
        pass

    def plt_chi_hist(self):
        pass