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

