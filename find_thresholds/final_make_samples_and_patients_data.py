
# coding: utf-8

# Notebook is created for vars: 
# 1. data_heal, data_sick (not shuffled), data_very_sick
# 2. patients_mean_heal, patients_mean_sick, patients_mean_very_sick

# In[10]:


get_ipython().run_line_magic('reload_ext', 'autoreload')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
    
from make_ready_data.make_ready_big_data import read_data, get_label, get_feat

from make_ready_data.make_ready_big_data import look_through_info, get_train_and_test_sets

from make_ready_data.make_ready_short_data import get_short_test_data_from_Roman

from make_ready_data.make_ready_short_data import get_short_test_data_from_Svetlana_G

from make_ready_data.create_features_indeces import get_labels_and_features_indeces

from make_ready_data.create_features_indeces import build_plot_for_best_features

from testing.testing_functions import testing_sick_patients,                     testing_sick_samples, extraxt_samples_from_new_test_data


# In[11]:


info, final_filelist = read_data(dirpath="../")

data_sick, data_heal, inds_sick, inds_heal = look_through_info(info, final_filelist, dirpath="../")

plot_labels, indlist = get_labels_and_features_indeces()

_, _, _, _, patients_data = get_train_and_test_sets(data_sick, data_heal, 
                                                               inds_sick, inds_heal, 0, 
                                                               balanced_data=False,
                                                               make_shuffle=False)

test_patients_data = get_short_test_data_from_Svetlana_G(dirpath='../data/test_data/')

test_patients_data.append(('Roman_patient', get_short_test_data_from_Roman("../data/short_test_data_from_Roman.csv")))


# In[12]:


# get mean vectors of 6 elements from group 1 and group 2
def extract_patients_mean_heal_and_sick(patients_data):
    patients_mean_heal = np.zeros((0, 6))
    patients_mean_sick = np.zeros((0, 6))
    for i,data in enumerate(patients_data):
        if data[0,-1] == 0:
            patients_mean_heal =np.concatenate((patients_mean_heal, data.mean(axis=0).reshape((1, -1))))
        else:
            patients_mean_sick =np.concatenate((patients_mean_sick, data.mean(axis=0).reshape((1, -1))))
    return patients_mean_heal, patients_mean_sick
patients_mean_heal, patients_mean_sick = extract_patients_mean_heal_and_sick(patients_data)


# In[13]:


# 92 samples from 18 very sick patients
def extract_data_very_sick(test_patients_data):
    data_very_sick = np.zeros((0, 5))
    for data in test_patients_data:
        data_very_sick =np.concatenate((data_very_sick, data[1]))
    ones = np.ones((data_very_sick.shape[0], 1))    
    data_very_sick = np.concatenate((data_very_sick, ones), axis=1)
    return data_very_sick
data_very_sick = extract_data_very_sick(test_patients_data)


# In[14]:


# get mean vector of 5 elements from group 3
def extract_patients_mean_very_sick(test_patients_data):
    patients_mean_very_sick = np.zeros((len(test_patients_data), 5))
    for i,data in enumerate(test_patients_data):
        patients_mean_very_sick[i] = data[1].mean(axis=0)
    return patients_mean_very_sick
patients_mean_very_sick = extract_patients_mean_very_sick(test_patients_data)

