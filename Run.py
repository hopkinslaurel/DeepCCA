#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import torch
import torch.nn as nn
import numpy as np
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler
from DeepCCAModels import DeepCCA
from main import Solver
from utils import load_data, svm_classify
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model_x', dest=model_x)
parser.add_argument('--model_y', dest=model_y)

torch.set_default_tensor_type(torch.DoubleTensor)


############
# Parameters Section

device = torch.device('cuda')
print("Using", torch.cuda.device_count(), "GPUs")

# the path to save the final learned features
save_to = './new_features.gz'

# the size of the new space learned by the model (number of the new features)
outdim_size = 64

# size of the input for view 1 and view 2
#input_shape1 = 784
#input_shape2 = 784

# number of layers with nodes in each one
layer_sizes1 = [1024, 1024, 1024, outdim_size]
layer_sizes2 = [1024, 1024, 1024, outdim_size]

# the parameters for training the network
learning_rate = 1e-3
epoch_num = 10
batch_size = 800

# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
reg_par = 1e-5

# specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
# if one option does not work for a network or dataset, try the other one
use_all_singular_values = False

# if a linear CCA should get applied on the learned features extracted from the networks
# it does not affect the performance on noisy MNIST significantly
apply_linear_cca = True
# end of parameters section
############

# Each view is stored in a csv file separately.
data1, input_shape1 = load_data(args.model_x)
data2,input_shape2 = load_data(args.model_y)


# Building, training, and producing the new features by DCCA
model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1,
                input_shape2, outdim_size, use_all_singular_values, device=device).double()
l_cca = None
if apply_linear_cca:
    l_cca = linear_cca()
solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                learning_rate, reg_par, device=device)
train1, train2 = data1[0][0], data2[0][0]
val1, val2 = data1[1][0], data2[1][0]
test1, test2 = data1[2][0], data2[2][0]
# val1=None
# test1=None
solver.fit(train1, train2, val1, val2, test1, test2)
# TODO: Save linear_cca model if needed

set_size = [0, train1.size(0), train1.size(
    0) + val1.size(0), train1.size(0) + val1.size(0) + test1.size(0)]
loss, outputs = solver.test(torch.cat([train1, val1, test1], dim=0), torch.cat(
    [train2, val2, test2], dim=0), apply_linear_cca)


# In[4]:


new_data = []
# print(outputs)
for idx in range(3):
    new_data.append([outputs[0][set_size[idx]:set_size[idx + 1], :],
                     outputs[1][set_size[idx]:set_size[idx + 1], :], data1[idx][1]])
# Training and testing of SVM with linear kernel on the view 1 with new features
[test_acc, valid_acc] = svm_classify(new_data, C=0.01)
print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
print("Accuracy on view 1 (test data) is:", test_acc*100.0)


# In[5]:


# Saving new features in a gzip pickled file specified by save_to
print('saving new features ...')
f1 = gzip.open(save_to, 'wb')
thepickle.dump(new_data, f1)
f1.close()


# In[6]:


d = torch.load('checkpoint.model')
solver.model.load_state_dict(d)
solver.model.parameters()


# In[ ]:




