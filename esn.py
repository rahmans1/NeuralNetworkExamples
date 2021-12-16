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

     """The time lag is taken to be the minimum of the first zero of acf and the first minimum of amf"""
    @property
    def tau(self):
        acf_tau = np.argwhere(self.acf<0)[0]-1
        amf_tau= np.array(signal.argrelextrema(self.amf,np.less))
        tau= np.array([acf_tau, amf_tau[0][0]])
        min_tau= np.min(tau.all())
        return min_tau

    """Calculate the False Nearest Neighbor Ratio for a default heuristic r upto maximum embedding dimension m"""
    def compute_m(self, r=5, max_m=10):
        s={}
        pairwise={}
        for m in range(1,max_m+2):
            s[m]=[]
            s[m].append(self.input)
            for i in range(m-1, 0,-1):
                s[m].append(np.roll(self.input, (m-i)*self.tau))
            s[m]=np.transpose(s[m])
            distance=cdist(s[m], s[m], metric='chebyshev')
            np.fill_diagonal(distance, np.nan)
            pairwise[m]=(distance[~np.isnan(distance)].reshape(distance.shape[0], distance.shape[1] - 1)).min(axis=1)

        sigma= np.std(self.input)
        for m in range(1, max_m+1):
            metric1=pairwise[m+1]/pairwise[m]-r
            metric1[metric1<0]=0
            metric1[metric1>0]=1
            metric2= sigma/r-pairwise[m]
            metric2[metric2<0]=0
            metric2[metric2>0]=1
            fnn_ratio=np.sum(metric1*metric2)/np.sum(metric2)
            plt.scatter(m, fnn_ratio)
        plt.yscale('log')
        plt.title("False Nearest Neighbor Ratio vs m (r="+str(r)+")")
        plt.show()
        return

    """Create input and output pairs from the whole data."""
    def create_input_output_pairs(self, m, k):

        input_array=[]
        input_array.append(self.input)
        for i in range(m-1, 0,-1):
            input_array.append(np.roll(self.input, (m-i)*self.tau))
        input_array=np.transpose(input_array)

        input  = input_array[(m-1)*self.tau+1:-k]
        output = self.input[(m-1)*self.tau+1+k:]

        return output, input

""" Class for echo state network """
class esn:
    def __init__(self, n_i, n_r, theta_ls, theta_sr, theta_rc):      
      """
      Inputs:
          theta_ls = input weights
          theta_sr = spectral radius of recurrent layer weight matrix
          theta_rc = sparsity of recurrent layer weight matrix
      """
      
      """ Initialize the recurrent layer weights """
      rng_r = np.random.default_rng(1234)
      uniform_r=stats.uniform(loc=-1, scale=2)
      uniform_r.random_state = np.random.RandomState(seed=5623)
      self.w_r= sparse.random(n_r, n_r, density=theta_rc, random_state=rng_r, data_rvs=uniform_r.rvs).todense()

      e, v = np.linalg.eig(self.w_r)
      max_e= np.max(np.abs(e))
      self.w_r*=  theta_sr/max_e

      """ Initialize the input weights  """
      rng_i = np.random.default_rng(2621)
      uniform_r=stats.uniform(loc=-1, scale=2)
      uniform_r.random_state = np.random.RandomState(seed=4322)
      self.w_i= sparse.random(n_r, n_i, density=1.0, random_state=rng_i, data_rvs=uniform_r.rvs).todense()
      self.w_i*= theta_ls
      self.w_o = None

      return


    def train(self,input,output):

      n_r = self.w_r.shape[0]
      X=[];
      X.append(np.asmatrix(np.zeros(n_r)));


      for i in range(1, input.shape[0]):
        A=X[i-1]@self.w_r.T
        B=input[i]@self.w_i.T
        X.append(np.tanh(A+B))

      X=np.array(X)[:,0,:]
      LHS=X.T@X+0.1*np.identity(n_r)
      RHS=X.T@output
      self.w_o = np.asmatrix(solve(LHS, RHS))

      return

    def predict(self,input):
      X=[];
      y=[]

      B=input[0]@self.w_i.T
      X.append(np.tanh(B))
      y.append(X[0]@self.w_o.T)
      for i in range(1, input.shape[0]):
        A=X[i-1]@self.w_r.T
        B=input[i]@self.w_i.T
        X.append(np.tanh(A+B))
        y.append(X[i]@self.w_o.T)

      return np.array(y).flatten()

def calculate_embedding(data):
    label= Path(data).stem
    print("Time series analysis for "+label+" data starts here")
    io_data = io(data)   # Load data
    plt.plot(io_data.acf, 'bo')    # Display the autocorrelation
    plt.xlim(0, 200)
    plt.title("Autocorrelation")
    plt.show()
    plt.plot(io_data.amf, 'bo')    # Display the auto mutual information
    plt.title("Auto Mutual Information")
    plt.show()
    io_data.compute_m(r=4)           
    # The heuristic is chosen arbitrarily. The embedding dimension is chosen to be the point where the FNN graph starts to plateau
    return io_data

    
def run_experiment(io_data, m, k, batch_size):
    output, input = io_data.create_input_output_pairs(m=m, k=k)  

    num_epochs=30                 
    end_batch= np.arange(0, input.shape[0], batch_size)
    end_batch= np.append(end_batch,input.shape[0])
    
    esn_model=esn(input.shape[1],200,0.3,0.9,0.4)
    train_input= None
    train_output= None

    
    for epoch in range(num_epochs):
      for i in range(1, end_batch.size-1):
          train_input = input[end_batch[0]:end_batch[i]]
          train_output = output[end_batch[0]:end_batch[i]]
          val_input  = input[end_batch[i]:end_batch[i+1]]
          val_output = output[end_batch[i]:end_batch[i+1]]
          esn_model.train(train_input,train_output)
          train_y=esn_model.predict(train_input)
          val_y= esn_model.predict(val_input)
          print("Epoch:"+str(epoch)+", Batch:"+str(i)+" Training RMSE:"+str(np.sqrt((np.square(train_y - train_output)).mean())))
          print("Epoch:"+str(epoch)+", Batch:"+str(i)+" Test RMSE:"+str(np.sqrt((np.square(val_y - val_output)).mean())))
    
    """
    plt.plot(train_output[0:200])
    plt.plot(train_y[0:200])
    plt.show()
    plt.plot(val_output[0:200])
    plt.plot(val_y[0:200])
    plt.show()
    """

    figure= plt.figure(figsize=(6000,75))
    ax = plt.subplots()
    ax[1].plot(train_output)
    ax[1].plot(train_y)
    ax[1].plot(np.arange(train_output.size,train_output.size+val_output.size), val_output)
    ax[1].plot(np.arange(train_output.size,train_output.size+val_output.size), val_y)
    plt.show()

    return
    
def main():
    #sin_data = calculate_embedding("/content/drive/MyDrive/AML_project_description/2sin.txt")
    #run_experiment(sin_data, m=4, k=10, batch_size=500)

    lorentz_data = calculate_embedding("/content/drive/MyDrive/AML_project_description/lorentz.txt")
    run_experiment(lorentz_data, m=4, k=10, batch_size=2400)




   # lorentz_data= calculate_embedding("/content/drive/MyDrive/AML_project_description/lorentz.txt")
    

if __name__ == "__main__":
    main()

