#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import sparse
from scipy import stats
from scipy import signal
from scipy.spatial.distance import cdist
from scipy.linalg import solve
# make sure plots are displayed correctly
%matplotlib inline


"""Class for dataset"""

class io:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, header=None)
        input = df[0].to_numpy()
        self.mean=np.mean(input)
        self.std=np.std(input)
        self.input= (input-self.mean)/self.std
        return

    """Calculate autocorrelation of data"""
    @property
    def acf(self):
        acf = np.correlate(self.input, self.input, mode='full')
        return acf[int(np.floor(acf.size/2)):]

    """Calculate automutual information of data"""
    @property
    def amf(self,bins=20, max_tau=200):
        amf=[]
        p_i, _ = np.histogram(self.input, bins, density=True)
        sum_p_i_log_p_i = np.nansum(p_i*np.log(p_i))
        for tau in range(0,max_tau):
          x = self.input[:self.input.size-tau]
          y = self.input[tau:]
          p_ij, _, _ = np.histogram2d(x, y, bins=[bins,bins], density=True)
          sum_p_ij_log_p_ij= np.nansum(p_ij*np.log(p_ij))
          amf.append(sum_p_ij_log_p_ij-sum_p_i_log_p_i)
        return np.array(amf)
