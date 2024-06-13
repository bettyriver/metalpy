#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:12:54 2024

@author: ymai0110
"""

import sys
sys.path.insert(0,'/project/blobby3d/Blobby3D/pyblobby3d/src/pyblobby3d/')
#from pyblobby3d.b3dcomp import setup_comparison_maps,map_limits,colorbar
#from pyblobby3d.b3dplot import plot_map,plot_colorbar,cmap
from b3dcomp import setup_comparison_maps,map_limits,colorbar
from b3dplot import plot_map,plot_colorbar,cmap

import numpy as np
import matplotlib.pyplot as plt

import dnest4 as dn4
#from pyblobby3d import PostBlobby3D
#from pyblobby3d import SpectralModel
from post_blobby3d import PostBlobby3D
from moments import SpectralModel
 
import matplotlib as mpl
from astropy.io import fits

