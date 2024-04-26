#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:50:12 2024

for pbs in Artemis

@author: ymai0110
"""

def time_calculator(pix):
    if pix<=300:
        time = pix/25
    elif pix<=400:
        time = 27*0.62
    elif pix<=500:
        time = 27*0.8
    elif pix<=800:
        time = 43
    elif pix<=1000:
        time = 110
    elif pix<=1310:
        time = 150
    else:
        time = 230
    return time

def cpu_mem(pix):
    if pix<=300:
        cpu, mem = 16,8
    elif pix<=400:
        cpu, mem = 18,13
    elif pix<=500:
        cpu, mem = 18,13
    elif pix<=800:
        cpu, mem = 19,16
    elif pix<=1000:
        cpu, mem = 20,18
    elif pix<=1310:
        cpu, mem = 20,22
    else:
        cpu, mem = 24,32
    return cpu, mem

def copy_postpy(data_path,id_list,save_path,save_name):
    
    '''create sh file that copy post.py to given id list
    
    data_path: path for data, e.g. data_v221
    id_list: list of magpiid, int64
    save_path: save path for this sh file
    save_name: file name for this sh file
    '''
    
    post_path = '/project/blobby3d/Blobby3dYifan/v150/post.py'    
    shfile = open(save_path+save_name,"w")
    
    
    for idd in id_list:
        shfile.write('cp '+post_path+' '+data_path+str(idd)+'/post.py\n')
    shfile.close()