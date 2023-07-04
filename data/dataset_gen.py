# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:55:01 2021

@author: Arshad
"""

import numpy as np
import pandas as pd


#load data

df = pd.read_csv('./data/ALL_EDs.dat', sep="\t", header=0, index_col=False)
df.rename(columns={"ED#":"ED"}, inplace=True)
df.dropna(inplace=True)
df.columns


#data size


print("total data size :", len(df))
for i in range(5, -1, -1): 
    print("DR%d size : %d"% (i,len(df.query("DR==%d"%i))))
    
    
    
#ED# no-ACK -> set SF12

import itertools

no_ack_end = []
# print("not ack index :", end=' ')
for ed in range(1, max(df['ED']) + 1, 1):
    #if np.count_nonzero(df.query('ED==%d'%ed)['ACK'].values) == 0:
    for i in range(1, max(df.query('ED==%d' % ed)['GroupID'].values), 1):
        no_ack_end.append(
            df.query("ED==%d and SF==12 and GroupID==%d" %
                     (ed, i)).index.values)

no_ack_end = list(itertools.chain(*no_ack_end))

for z in no_ack_end:
    df.loc[z, 'ACK'] = 1
    
    
#select lowest SF based on the ACK and generate sequence of the data

def generate_sequence_dataset(x_data, y_data, sequence=20):
    x_data_list = []
    y_data_list = []
    x_data_length = len(x_data)
    for i in range(x_data_length-sequence):
        x_data_list.append(x_data[i:i+sequence])
        y_data_list.append(y_data[i+sequence:i+sequence+1])
    return np.array(x_data_list), np.array(y_data_list)


# end device == 500
# group_id == ucnt(uplink counter) == 6
tmp = []
for ed in range(1, max(df['ED']) + 1, 1):
    df_ed_n = df.query('ED==%d' % ed)
    for ucnt in range(1, max(df_ed_n['GroupID']), 1):

        sf = df_ed_n.query('GroupID==%d' % ucnt)['SF'].values
        ack = df_ed_n.query('GroupID==%d' % ucnt)['ACK'].values
        sf_ack = sf * ack

        try:
            best_sf = np.min(sf_ack[sf_ack > 0])
        except:
            best_sf = 0
        #ed=df_ed_n.query('GroupID==%d and SF==%d'%(ucnt, best_sf))['ED'].values[0]
        group = df_ed_n.query('GroupID==%d and SF==%d' %
                              (ucnt, best_sf))['GroupID'].values[0]
        x_pos = df_ed_n.query('GroupID==%d and SF==%d' %
                              (ucnt, best_sf))['X-pos'].values[0]
        y_pos = df_ed_n.query('GroupID==%d and SF==%d' %
                              (ucnt, best_sf))['Y-pos'].values[0]
        dist = df_ed_n.query('GroupID==%d and SF==%d' %
                             (ucnt, best_sf))['Distance'].values[0]
        rx_pw = df_ed_n.query('GroupID==%d and SF==%d' %
                              (ucnt, best_sf))['RxPw'].values[0]
        snr = df_ed_n.query('GroupID==%d and SF==%d' %
                            (ucnt, best_sf))['SNR'].values[0]
        snr_req = df_ed_n.query('GroupID==%d and SF==%d' %
                                (ucnt, best_sf))['SNR_req'].values[0]

        print([ed, group, x_pos, y_pos, rx_pw, snr, snr_req, best_sf])

        tmp.append(
            np.array(
                [ed, group, x_pos, y_pos, dist, rx_pw, snr, snr_req, best_sf]))
    #break
np_tmp = np.array(tmp)


#generate sequence data

np.save("./data/dataset_test", np_tmp)
dataset = np.load("./data/dataset_test.npy")

def generate_sequence_dataset(x_data, y_data, sequence=6):
    x_data_list = []
    y_data_list = []
    x_data_length = len(x_data)
    for i in range(x_data_length - sequence):
        x_data_list.append(x_data[i:i + sequence])
        y_data_list.append(y_data[i + sequence:i + sequence + 1])
    return np.array(x_data_list), np.array(y_data_list)


df_ = pd.DataFrame(dataset,
                   columns=[
                       'ED', 'GroupID', 'X-pos', 'Y-pos', 'Distance', 'RxPw',
                       'SNR', 'SNR_req', 'Best_SF'
                   ])


x_data = df_.query("ED==1").drop(['ED', 'GroupID','Best_SF'],axis=1).values
y_data = df_.query("ED==1")['Best_SF'].values

x_data_np, y_data_np = generate_sequence_dataset(x_data, y_data)

for h in range(2,500+1,1):
    x_data_2 = df_.query("ED==%d"%h).drop(['ED', 'GroupID','Best_SF'],axis=1).values
    y_data_2 = df_.query("ED==%d"%h)['Best_SF'].values
    
    x, y= generate_sequence_dataset(x_data_2, y_data_2)
    x_data_np = np.append(x_data_np, x, axis=0)
    y_data_np = np.append(y_data_np, y, axis=0)
    
    print(h)
    
print(x_data_np.shape)
print(y_data_np.shape)
