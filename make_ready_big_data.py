
# coding: utf-8

# ### Генерим новые таблицы
# Все очень просто. Есть тренировочные пациенты и тестируемые пациенты.
# Так вот увеличивать данные будем так же в разных группах независимо, чтоб обучаясь на одном пациенте, классфикаторы не выдавали верный ответ только потому, что встретили знакомого пациента.
# 
# Подробнее. 
# 
# Остановимся на особенностях нотбука: 
# 1. Для работы нужны только директория 'meas' и файл 'info.csv'. Они лежат в папке 'data'. 
# 2. К сожалению названия методов мало что говорят.
# 3. нотбук создан после того как определились, что нужны только признаки RR, TQ, QTc,JTc, Tp-Te
# 
# <br>plot_labels = ['RR', 'TQ', 'QTc', 'JTc', 'TpeakTend'] ------> WRONG!
# <br>plot_labels = ['QTc', 'JTc', 'TpeakTend', 'TQ', 'RR'] ------> CORRECT!
# 
# В конце вызываются подряд три метода:
# 1.  read_data(); returns:
#     1. info  ------------------- просто DataFrame с 'info.scv'; 
#     2. final_filelist ---------- перечень тех файлов которые есть у нас и описаны в 'info.scv'
# 
# 2.  look_through_info(info, final_filelist); returns:
#     1. data_sick, data_heal ---- данные, где каждая строка - паттерн (иначе сэмпл), колонки - 5 признаков+label 
#     2. inds_sick, inds_heal ---- строк сколько больных/здоровых пациентов. 2 колонки, во второй - сколько сэпплов в файле пациента, в первой - cumsum второй колонки (нужно для того, чтобы затем найти в data_sick, откуда сэмплы начинаются для этого пацента и где заканчиваются, метод: get_data_by_indexes)
# 
# 3.  get_train_and_test_sets(data_sick, data_heal, inds_sick, inds_heal, per_edge)
#     1. tr_d, ts_d -------------- уже готовые данные
#     2. tr_l, ts_l -------------- ответы к ним

# In[1]:


import pandas as pd

import numpy as np

import os


# In[2]:


def read_data():
    info = pd.read_csv('data/info.csv')

    # astype(int) for getting rid of '123123.0' and strings for next step
    file_numbers = info['Номер ЭКГ'].dropna().astype(int).astype(str)

    info_filelist = 'results' + file_numbers + '.csv'

    meas_filelist = os.listdir('data/meas')

    # form intersection of our 2 sets
    final_filelist = set(info_filelist).intersection(meas_filelist)
    
    return info, final_filelist


# In[3]:


# l = [RR, TQ, QTc, JTc, TpeakTend]
# l = [QTc, JTc, TpeakTend, TQ, RR]

def get_label(info, filename):

    #     find the row with our name of file
    row = info.loc[info['Номер ЭКГ'].isin([filename[7:-4]])]
    
    #     extract the and label of our file
    answer = row['сердечно-сосудистое заболевание (при наличии)'].all()
    
    return (1 if answer == 'да' else 0)

def get_feat(arrs) :
    shiftedRR = np.sqrt(arrs[0,:-1].copy())
    #     arrs.shape[1]-1 ---> -1 cause of first RR
    newarrs = np.empty((5+1, arrs.shape[1]-1), dtype=float)
    
    newarrs[0,:] = arrs[2,1:] / shiftedRR                   #QTc = QT / shiftedRR
    newarrs[1,:] = arrs[1,1:] / shiftedRR                   #JTc = JT / shiftedRR         
    newarrs[2,:] = arrs[3,1:]                               #TpeakTend = TT
    newarrs[3,:] = arrs[0,1:] - arrs[2,1:]                  #TQ = RR – QT
    newarrs[4,:] = arrs[0,1:]                               #RR

    return newarrs


# In[4]:


def look_through_info(info, final_filelist):
    
    def fill_data(data, inds, ii, arr):
        data = np.concatenate((data, arr.T))
        inds[ii,0] = inds[ii-1,0] + inds[ii-1,1] # if ii == 0 then take inds[-1,0] == 0
        inds[ii,1] = arr.shape[1]
        return data, ii+1

    len_list = len(final_filelist) # number of all files
    
    #     data of patients with label
    data_sick = np.empty((0, 5 + 1), dtype=float)
    data_heal = np.empty((0, 5 + 1), dtype=float)
    
    #     cumsum in 0-column and number of patterns in 1-column
    inds_sick = np.zeros((len_list, 2), dtype=int)
    inds_heal = np.zeros((len_list, 2), dtype=int)
    i_sick = 0
    i_heal = 0
            
    for filename in final_filelist :
        df = pd.read_csv('data/meas/' + filename, 
                         skiprows=list(range(10))+[11, 15, 16]+list(range(17,99)), 
                         index_col=None, header=None)
        data = df._get_numeric_data()
        if data.shape[1] <= 1: # because RR will shorted by one
            continue
        arr = get_feat(data.values)
        ans = get_label(info, filename)
        arr[5] = ans
        if ans == 1:
            data_sick, i_sick = fill_data(data_sick, inds_sick, i_sick, arr)
        else:
            data_heal, i_heal = fill_data(data_heal, inds_heal, i_heal, arr)
    inds_sick = inds_sick[:i_sick]
    inds_heal = inds_heal[:i_heal]
    
    return data_sick, data_heal, inds_sick, inds_heal


# In[28]:


def get_train_and_test_sets(DataSick, DataHeal, IndsSick, IndsHeal, per_edge=0.8):
        
    data_sick = DataSick.copy()
    data_heal = DataHeal.copy()
    inds_sick = IndsSick.copy() 
    inds_heal = IndsHeal.copy()
    
    #     number_of_health -- number of health patterns
    #     take exactly helth because its less than sick 
    number_of_health = inds_heal[-1,0] + inds_heal[-1,1]
    edge1 = int(number_of_health * per_edge) # how much is train patterns
    edge2 = number_of_health
    def get_data_by_indexes(ready_data, inds):
        data = np.empty((0,5+1))
        for i in range(inds.shape[0]):
            start = inds[i,0]
            end = start + inds[i,1]
            data = np.concatenate((data, ready_data[start:end]))
        return data
        
    def get_train_test_data(ready_data, inds):
        np.random.shuffle(inds)
        ind_cumsum = np.cumsum(inds[:,1])
        ind_train = inds[ind_cumsum < edge1]
        ind_test = inds[(ind_cumsum >= edge1) * (ind_cumsum<edge2)]
        train_data = get_data_by_indexes(ready_data, ind_train)
        test_data = get_data_by_indexes(ready_data, ind_test)
        return train_data, test_data
    
    
    
    tr_s, ts_s =  get_train_test_data(data_sick, inds_sick)
    tr_h, ts_h = get_train_test_data(data_heal, inds_heal)
        
    train = np.concatenate((tr_s,tr_h))
    test = np.concatenate((ts_s,ts_h))

    np.random.shuffle(train)
    np.random.shuffle(test)
    return train[:,:5], test[:,:5], train[:,5], test[:,5]


# In[29]:


info, final_filelist = read_data()
data_sick, data_heal, inds_sick, inds_heal = look_through_info(info, final_filelist)
tr_d, ts_d, tr_l, ts_l = get_train_and_test_sets(data_sick, data_heal, inds_sick, inds_heal, 0.9)

