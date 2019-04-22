
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import matplotlib.pyplot as plt
import math
ds = pd.read_csv('Admission_Predict.csv') 
x = ds[['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA']]

y = ds.iloc[:,-1]
M = mean(x.T,axis=1)
print M
c = x-M
V = cov(c.T)

eig_vals,eig_vecs = eig(V)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
exp_var_percentage = 0.97 # Threshold of 97% explained variance

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

num_vec_to_keep = 0

for index, percentage in enumerate(cum_var_exp):
  if percentage > exp_var_percentage:
    num_vec_to_keep = index + 1
    break
num_features = x.shape[1]
proj_mat = eig_pairs[0][1].reshape(num_features,1)
for eig_vec_idx in range(1, num_vec_to_keep):
  proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))

# Project the data 
pca_data = x.dot(proj_mat)
print(pca_data)

