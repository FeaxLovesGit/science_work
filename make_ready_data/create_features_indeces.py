
# coding: utf-8

# 20 июля 2018
# 
# <h5> Создаем функцию get_labels_and_features_indeces(), которая возвращает: </h5>
# <br> правильную последовательность названий столбцов
# <br> лист из всевозможных комбинаций индексов признаков
# 
# 
# <h5> Создаем функцию build_plot_for_best_features(...), которая: </h5>
# <br> строит график с результатами по всем признакам

# In[4]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


# In[5]:


def get_labels_and_features_indeces():
    plot_labels = ['QTc', 'JTc', 'TpeakTend', 'TQ', 'RR']
    N_FEATURES = len(plot_labels)
    indlist = []
    for i in range(1, 2**N_FEATURES):
        indlist_tmp = []
        bin_str = '{0:b}'.format(i)
        bin_str = '0'*(N_FEATURES-len(bin_str)) + bin_str

        for j,c in enumerate(bin_str):
            if c=='1':
                indlist_tmp.append(N_FEATURES-j-1)
        indlist.append(indlist_tmp)
        
    return plot_labels, indlist


# In[11]:


def build_plot_for_best_features(plot_labels, indlist, y,
                                 y_arange=np.arange(0.55,0.75,0.01),
                                 marker=[':ro'],
                                 legend_label=['LogisticRegression: C = 1000'], 
                                 savefile_name='best_features.png'):
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (15,10)
    x = np.arange(len(indlist))

    for i in range(len(y)):
        plt.plot(x, y[i], marker[i], label=legend_label[i])

    plt.legend(fontsize=15)

    plt.xticks(range(len(indlist)), [str(np.array(plot_labels)[indlist[i]])                                         for i in range(len(indlist))], fontsize=9, rotation=90)
    plt.yticks(y_arange, fontsize=15)

    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.savefig(savefile_name, bbox_inches='tight')
    plt.show()


# In[14]:


# plot_labels, indlist = get_labels_and_features_indeces()
# y = [np.linspace(0.55,0.75,len(indlist))]
# build_plot_for_best_features(plot_labels, indlist, y)

