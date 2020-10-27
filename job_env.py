# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:07:58 2020

@author: lvjf

job env for the project
"""

import numpy as np
import pandas as pd

class job_shop_env():
    path = 'D:/spyderwork/Statistical_Learning/project/data/'
    expert_job = pd.read_csv(path + 'process_time_matrix.csv',header=None).drop([0]).values
    job = pd.read_csv(path + 'work_order.csv',header=None).values
    
    def __init__(self):
        self.job_cluster = self.expert_job.shape[1]
        self.expert = self.expert_job.shape[0]
        self.job_num = self.job.shape[0]
        self.process_time = self.expert_job
        self.expert_status = np.repeat(0,self.expert) ## how many jobs an expert is processing
        self.expert_process_job = [[] for i in range(self.expert)]
        self.expert_process_time = [[] for i in range(self.expert)]
        self.job_waiting_time = [[] for i in range(self.expert)]
        self.left_job = self.job.shape[0]
        self.done = False
        self.total_time = 0  ## total process time
        self.job_distribute_time = np.repeat(0,self.job.shape[0])
        self.total_job_process_time = np.repeat(0,self.job.shape[0])
        self.job_status = np.repeat(1,self.job.shape[0])  ## whether a job is under process
        self.job_index = list(range(self.job.shape[0]))  ## use for sampling
        self.timeindex = 0   ## use for time recording
        self.state = np.hstack((self.expert_status,self.job_distribute_time))
        self.done_job = [] ## how many jobs have been done
        
        
    def reset(self):
        self.job_num = 1
        self.expert_status = np.repeat(0,self.expert) ## how many jobs an expert is processing
        self.expert_process_job = [[] for i in range(self.expert)]
        self.expert_process_time = [[] for i in range(self.expert)]
        self.job_waiting_time = [[] for i in range(self.expert)]
        self.left_job = self.job.shape[0]
        self.done = False
        self.total_time = 0  ## total process time
        self.job_distribute_time = np.repeat(0,self.job.shape[0])
        self.total_job_process_time = np.repeat(0,self.job.shape[0])
        self.job_status = np.repeat(1,self.job.shape[0])  ## whether a job is under process
        self.job_index = list(range(self.job.shape[0]))  ## use for sampling
        self.timeindex = 0   ## use for time recording
        self.state = np.hstack((self.expert_status,self.job_distribute_time))
        self.done_job = []
        
    def step(self, action):
        # random generate job
        job_id = np.random.choice(a=self.job_num, size=self.expert, replace=False, p=None)
        for i in job_id:
            if i in self.job_index:
                self.job_distribute_time[i] += 1
                ## if more than 5, delete this job
                if self.job_distribute_time[i] >= 5:
                    del self.job_index[self.job_index.index(i)]
        
        assert action.shape[0] == self.expert
        
        for i in range(self.expert):
            ## only process those jobs that are in job_index
            if job_id[i] in self.job_index:
                ## action = 0 indicates do not give jobs to the expert
                if action[i] == 0 or self.expert_status[i] == 3:
                    pass
                else:
                    self.expert_process_job[i].append(job_id[i])
                    self.expert_status[i] += 1
                    self.expert_process_time[i].append(0)
                    # how much time a job wait before processing
                    self.job_waiting_time[i].append(self.timeindex)
                    self.total_job_process_time[job_id[i]] = self.process_time[i][job_id[i]]
                
                for j in len(self.expert_process_time[i]):
                    if self.expert_process_time[i][j] == self.process_time[i][self.expert_process_job[j]]:
                        # if job finished, workload of expert would decrease
                        self.expert_status[i] -= 1
                        if self.expert_process_job[j] not in self.done_job:
                            self.left_job -= 1
                        self.done_job.append(self.expert_process_job[j])
            ## calculate total time consumed
            self.total_time += sum(self.job_waiting_time[i]) + sum(self.total_job_process_time[i])
        
        ## reward takes the minus of total time*0.001
        reward = -0.001*self.total_time
        self.timeindex += 1
        self.expert_process_time += 1
        ## update state info
        self.state = np.hstack((self.expert_status,self.job_distribute_time))
        
        if self.left_job == 0:
            self.done = True
        return self.state, reward, self.done