
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


# In[68]:


def get_short_test_data_from_Romanov(filepath="./test_data.ods", sep='delimiter',  automatic_measure=True):
    df = pd.read_csv(filepath, header=None, index_col=0)
    if automatic_measure:
        print(df.index[4] + "is dropped")
        df = df.drop(df.index[4])
    else:
        print(df.index[3] + "is dropped")
        df = df.drop(df.index[3])
    return df.values.T


# In[69]:


# get_short_test_data_from_Romanov("./data/short_test_data_from_Roman.csv", automatic_measure=False)

