#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
from blobby3d by mat

@author: mat
"""

from pathlib import Path
import numpy as np


class Metadata:

    def __init__(self, metadata_path):
        """Metadata oranisation object.

        Parameters
        ----------
        metadata_path : str or pathlib.Path

        Returns
        -------
        None.

        Attributes
        ----------
        sz : float
        x_lim : (float, float)
            Left and right most boundaries of the x axis.
        y_lim : float
            Bottom and top most boundaries of the y axis.
        r_lim : float
            Left and right most boundaries of the wavelength axis.
        dx : float
            Width of pixels along the x axis.
        dy : float
            Width of pixels along the y axis.
        dr : float
            Width of pixels along the wavelength axis.

        """
        metadata = np.loadtxt(Path(metadata_path))
        self.naxis = metadata[:3].astype(int)
        self.sz = self.naxis.prod()
        self.x_lim = metadata[3:5]
        self.y_lim = metadata[5:7]
        self.r_lim = metadata[7:9]
        self.dx = np.diff(self.x_lim)[0]/self.naxis[1]
        self.dy = np.diff(self.y_lim)[0]/self.naxis[0]
        self.dr = np.diff(self.r_lim)[0]/self.naxis[2]
