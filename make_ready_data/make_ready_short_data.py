
# coding: utf-8

# Просто получили массив значений. Каждая строка - отдельный пациент.
# 
# Признаки в столбцах согласно корректному plot_labels. 
# <br>plot_labels = ['RR', 'TQ', 'QTc', 'JTc', 'TpeakTend'] ------> WRONG!
# <br>plot_labels = ['QTc', 'JTc', 'TpeakTend', 'TQ', 'RR'] ------> CORRECT!

# In[1]:


import pandas as pd

import numpy as np

import os


# In[2]:


def get_short_test_data_from_Roman(filepath="./data/short_test_data_from_Roman.csv", automatic_measure=True):
    df = pd.read_csv(filepath, header=None, index_col=0)
    if automatic_measure:
        print(df.index[4] + "is dropped")
        df = df.drop(df.index[4])
    else:
        print(df.index[3] + "is dropped")
        df = df.drop(df.index[3])
    return df.values.T


# In[13]:


# get_short_test_data_from_Roman("../data/short_test_data_from_Roman.csv", automatic_measure=False)


# In[4]:


def get_short_test_data_from_Svetlana_G(dirpath="../data/test_data"):
    plot_labels = ['QTc(B)', 'JTc(B)', 'TpeakTend', 'TQ', 'R-R']
    list_files = os.listdir(dirpath)
    test_patient_data_list = []
    for file in list_files:
        df = pd.read_csv(dirpath+"/"+file, header=None,
                         skiprows=[0,2,3,4,6]+list(range(10, 20)), index_col=1)
        
        df = df[df < 3].dropna(axis=1)
        df = df.reindex(plot_labels)
        test_patient_data_list.append((file, df.values.T))
    return test_patient_data_list


# In[5]:


# get_short_test_data_from_Svetlana_G(dirpath="/home/feax/Desktop/FeaxLovesGit/science_work/data/test_data")

def extraxt_samples_from_new_test_data(test_patients_data):
    patients_samples = np.empty((0, 5))
    for data in test_patients_data:
        patients_samples = np.concatenate((patients_samples, data[1]))
    return patients_samples