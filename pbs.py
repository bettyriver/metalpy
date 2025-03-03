#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:50:12 2024

for pbs in Artemis

@author: ymai0110
"""

def time_calculator(pix,emi_line):
    
    if emi_line=='Ha':
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
    else:
        if pix<=300:
            time = pix/25
        elif pix<=400:
            time = 27*0.62
        elif pix<=500:
            time = 27*0.8
        elif pix<=800:
            time = 43
        elif pix<=1000:
            time = 100 # lower iterations
        elif pix<=1310:
            time = 120 # lower iterations
        else:
            time = 180 # lower iterations
        
        time = time/2 + 4 # smaller wavelength range
    
    return time

def cpu_mem(pix,emi_line):
    '''larger cpu and memory for Halpha,
        smaller cpu and memory for other lines, e.g. Oii, Hb
    
    '''
    
    if emi_line=='Ha':
    
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
    
    else:
    
        if pix<=300:
            cpu, mem = 14,2
        elif pix<=400:
            cpu, mem = 16,3
        elif pix<=500:
            cpu, mem = 16,5
        elif pix<=800:
            cpu, mem = 17,10
        elif pix<=1000:
            cpu, mem = 18,10
        elif pix<=1310:
            cpu, mem = 18,14
        else:
            cpu, mem = 22,16
        
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


def pbs_run_b3d(pbs_path,pbs_name,pbs_file_num,id_pix_sorted,data_path,b3d_path):
    '''
    pbs_path: path to write pbs file
    pbs_name: name of pbs file
    pbs_file_num: to split full sample to how many pbs files
    id_pix_sorted: 2 column array, first column is id, second column is number 
                    of pixels. sorted according to pixel number ascending
    data_path: path that save preblobby3d file (data.txt, var.txt...)
    b3d_path: path for blobby3d exe
    
    
    
    '''
    flag = 0
    sum_time = 0
    file_num = 1
    for flag in range(pbs_file_num):
        pbsfile = open(pbs_path+pbs_name+"_"+str(file_num)+'.pbs',"w")
        str_to_write = ''
        #while True:
        sum_time += time_calculator(id_pix_sorted[flag][1])
        str_to_write += 'cd '+data_path+str(int(id_pix_sorted[flag][0]))+'\n'
        str_to_write += 'export OMP_NUM_THREADS=32\n'
        str_to_write += '${io}Blobby3D -t 32 -f MODEL_OPTIONS\n'
            
        max_pix = id_pix_sorted[flag][1]
            
        cpu, mem = cpu_mem(max_pix)
        pbsfile.write('#!/bin/bash\n')
        pbsfile.write('#PBS -P blobby3d\n')
        pbsfile.write('#PBS -N metal_test_con_pa_'+str(file_num)+'\n')
        pbsfile.write('#PBS -l select=1:ncpus=%d:mem=%dGB\n'%(cpu,mem))
        if file_num ==7:
            run_time = int((sum_time+8)*0.7)
            pbsfile.write('#PBS -l walltime=%d:00:00\n'%(int((sum_time+8)*0.7)))
        else:
            run_time = int((sum_time+8)/2)
            pbsfile.write('#PBS -l walltime=%d:00:00\n'%(int((sum_time+8)/2)))
        pbsfile.write('#PBS -q defaultQ\n')
        pbsfile.write('io='+b3d_path+'\n')
        pbsfile.write(str_to_write)
        pbsfile.close()
        print(str(file_num)+': time:'+str(run_time)+' h')
        sum_time = 0
        file_num += 1