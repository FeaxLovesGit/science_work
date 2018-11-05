
# coding: utf-8

# In[4]:


import numpy as np

from sklearn.linear_model import LogisticRegression


# In[5]:


def extraxt_samples_from_new_test_data(test_patients_data):
    patients_samples = np.empty((0, 5))
    for data in test_patients_data:
        patients_samples = np.concatenate((patients_samples, data[1]))
    return patients_samples


# In[6]:


def testing_sick_samples(data_sick, data_heal, indlist, test_patients_samples):
    all_data = np.concatenate((data_heal, data_sick))
    res = np.zeros(len(indlist))
    for i, inds in enumerate(indlist):
        clf = LogisticRegression(C=1000.)
        clf.fit(all_data[:, inds], all_data[:, -1])
        res[i] = clf.predict(test_patients_samples[:, inds]).sum() / test_patients_samples.shape[0]
    return res


# In[7]:


def testing_sick_patients(data_sick, data_heal, indlist, test_patients_data, C=1000, print_log=True):
    all_data = np.concatenate((data_heal, data_sick))
    res_for_patients = np.zeros(len(indlist))
    for i, inds in enumerate(indlist):
        if print_log:
            print(inds)
        clf = LogisticRegression(C=C)
        clf.fit(all_data[:, inds], all_data[:, -1])
        for j in range(len(test_patients_data)):
            res = clf.predict(test_patients_data[j][1][:, inds])
            log_ans = round(res.sum() / res.shape[0] + 0.0001)
            if print_log:
                print(test_patients_data[j][0], int(log_ans), res)
            res_for_patients[i] += log_ans
        if print_log:
            print()
    res_for_patients = res_for_patients / len(test_patients_data)
    return res_for_patients

