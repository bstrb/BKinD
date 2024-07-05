#integrate.py

# Standard library imports
import sys

# Third-party imports
import numpy as np
import pandas as pd

# CCTBX imports
from iotbx.reflection_file_reader import any_reflection_file
from iotbx.shelx.hklf import miller_array_export_as_shelx_hklf as hklf
from scitbx.array_family import flex

class Integrate():
    
    def __init__(self, file_name):
        """ 
        Class for easy access and manipulation of INTEGRATE.HKL file. 
        
        Args:
            file_name: path to INTEGRATE.HKL file
        """
        self.inte= any_reflection_file(file_name)\
            .as_miller_arrays(merge_equivalents=False)
        self.inte0 = self.inte[0]
        # print(self.inte0)
        self.inte1 = self.inte[1]
        # print(self.inte1)
        self.inte2 = self.inte[2]
        # print(self.inte2)
        self.df = pd.DataFrame()
        
    def set_resolution(self, dmax: float, dmin: float):
        """ 
        Resoltion filter
        y
        
        Args:
            dmax: maximum resolution
            dmin: minimum resolution
        
        """
        self.inte0 = self.inte[0].resolution_filter(dmax, dmin)
        self.inte1 = self.inte[1].resolution_filter(dmax, dmin)
        self.inte2 = self.inte[2].resolution_filter(dmax, dmin)
    
    def indices(self):
        """ 
        Returns indices
        """
        return(list(self.inte0.indices()))
    
    def hex_bravais(self):
        h = np.array(self.indices())[:,0]
        k = np.array(self.indices())[:,1]
        l = np.array(self.indices())[:,2]
        i = -(h+k)
        
        return list(zip(h, k, i, l))
    
    def data(self):
        """ 
        Returns intensities
        """
        return list(self.inte0.data())
    
    def sigmas(self):
        """ 
        Returns sigma(intensity)
        """
        return list(self.inte0.sigmas())
    
    def xobs(self):
        """ 
        Returns observed frame number
        """
        return np.array(self.inte1.data())[:, 0]
    
    def yobs(self):
        """ 
        Returns observed frame number
        """
        return np.array(self.inte1.data())[:, 1]
    
    def zobs(self):
        """ 
        Returns observed frame number
        """
        return np.array(self.inte1.data())[:, 2]
    
    def d_spacings(self):
        """ 
        Returns resolution
        """
        return list(self.inte0.d_spacings().data())
    
    def asus(self):
        """ 
        Returns asymmetric group
        """
        return list(self.inte0.map_to_asu().indices())

    def as_df(self) -> pd.DataFrame:
        """ 
        Converts into a DataFrame
        """
        self.df = pd.DataFrame()
        self.df['Miller'] = self.indices()
        self.df['asu'] = self.asus()
        self.df['Intensity'] = self.data()
        self.df['Sigma'] = self.sigmas()
        self.df['I/Sigma'] = self.df['Intensity']/self.df['Sigma']
        self.df['Resolution'] = self.d_spacings()
        self.df['xobs'] = self.xobs()
        self.df['yobs'] = self.yobs()
        self.df['zobs'] = self.zobs()
        self.df['Index_INTE'] = np.arange(0, len(self.df.index), 1)
        
        return self.df

    def as_df_hex(self):
        self.as_df()
        self.df['brav'] = self.df['Miller'].apply(
            lambda x: (x[0], x[1], -(x[0]+x[1]), x[2]))
        column_order = ['Miller', 'brav', 'asu', 'Intensity', 'Sigma',
                        'I/Sigma', 'Resolution', 'xobs', 'yobs', 'zobs', 
                        'Index_INTE']
        self.df = self.df.reindex(columns=column_order)
        
        return self.df
        
    def sele_idx(self, sel: np.array):
        """ 
        Select from INTEGRATE.HKL given corresponding indices
        sel = np.asarray(df.loc[df['Column'] > condition]['Index_INTE'])
        """
        sel = flex.size_t(sel)
        self.inte0 = self.inte0.select(sel)
        self.inte1 = self.inte1.select(sel)
        self.inte2 = self.inte2.select(sel)
    
    def sele_idx_as_df(self, sel: np.ndarray) -> pd.DataFrame:
        self.sele_idx(sel)
        
        return self.as_df()
    
    def sele_bool(self, sel: np.ndarray):
        """ 
        Select from INTEGRATE.HKL given Boolean indices
        sel = np.asarray(df['Column'] > condition)
        """
        sel = flex.bool(sel)
        self.inte0 = self.inte0.select(sel)
        self.inte1 = self.inte1.select(sel)
        self.inte2 = self.inte2.select(sel)
        
    def sele_bool_as_df(self, sel: np.ndarray) -> pd.DataFrame:
        self.sele_bool(sel)
        
        return self.as_df()
    
    def print_hklf4(self, outp_name: str):
        """ 
        Converts iotbx.any_reflection_file integratehkl to SHELX HKLF4 format 
        and output to outp_name.hkl
        
        Args:
            path_to_integrate_hkl (str): a path to INTERGATE.HKL file
            outp_name (str): desired output filename
        """
        # Process INTEGRATE.HKL 
        stdout_obj = sys.stdout
        sys.stdout = open(outp_name+'.hkl', 'w')
        hklf(self.inte0)
        sys.stdout = stdout_obj

    def print_hklf4_df(self, outp_name: str):

        self.print_hklf4()