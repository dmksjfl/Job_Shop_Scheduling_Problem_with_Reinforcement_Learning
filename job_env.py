# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:07:58 2020

@author: lvjf

job env for the project
"""

import numpy as np
import pandas as pd
import random

class job_shop_env():
    path = './data/'
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
        self.state = np.vstack((self.job_status,self.job_distribute_time))
        self.state = self.state.reshape(self.state.shape[0],self.state.shape[1],1)
        self.done_job = [] ## how many jobs have been done
        self.done_expert = [] ## which expert compelete the job corresponds to done_job list
        self.job_start_time = [] ## when do the job start to be processed, used in final result generation
        self.state_dim = self.state.shape[0]
        self.action_dim = 2
        
        
    def reset(self):
        self.job_num = self.job.shape[0]
        self.expert_status = np.repeat(0,self.expert) ## how many jobs an expert is processing
        self.expert_process_job = [[] for i in range(self.expert)]
        self.expert_process_time = [[] for i in range(self.expert)]
        self.job_waiting_time = [[] for i in range(self.expert)]
        #self.left_job = self.job.shape[0]
        self.done = False
        self.total_time = 0  ## total process time
        self.job_distribute_time = np.repeat(0,self.job.shape[0])
        self.total_job_process_time = np.repeat(0,self.job.shape[0])
        self.job_status = np.repeat(1,self.job.shape[0])  ## whether a job is under process
        self.job_index = list(range(self.job.shape[0]))  ## use for sampling
        #self.timeindex = 0   ## use for time recording
        self.state = np.vstack((self.job_status,self.job_distribute_time))
        self.state = self.state.reshape(self.state.shape[0],self.state.shape[1],1)
        self.done_job = []
        self.done_expert = [] ## which expert compelete the job corresponds to done_job list
        self.job_start_time = [] ## when do the job start to be processed, used in final result generation
        
        return self.state
        
    def step(self, action):
        # random generate job
        job_id = np.random.choice(a=self.job_num, size=self.expert, replace=False, p=None)
        for i in job_id:
            if len(self.job_index) != 0:
                if i in self.job_index:
                    self.job_distribute_time[i] += 1
                    ## if more than 2, delete this job
                    #if self.job_distribute_time[i] >= 2:
                    #    del self.job_index[self.job_index.index(i)]
                else:
                    job_id[job_id.tolist().index(i)] = random.sample(self.job_index,1)[0]
            else:
                pass
        
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
                    self.job_status[job_id[i]] = 0
                    self.expert_process_time[i].append(0)
                    # how much time a job wait before processing
                    self.job_waiting_time[i].append(self.timeindex)
                    # if expert could not handle the job, exit
                    self.total_job_process_time[job_id[i]] = self.process_time[i][self.job[job_id[i]][2]]
                
                delete_index = []
                for j in range(len(self.expert_process_time[i])):
                    if len(self.expert_process_job[i]) != 0:
                        if self.expert_process_time[i][j] == self.process_time[i][self.job[self.expert_process_job[i][j]][2]]:
                            # if job finished, workload of expert would decrease
                            self.expert_status[i] -= 1
                            self.done_expert.append(i)
                            if self.expert_process_job[i][j] not in self.done_job:
                                self.left_job -= 1
                            self.done_job.append(self.expert_process_job[i][j])
                            ## calculate when the job starts to be processed by subtracting the process time
                            self.job_start_time.append(self.job_waiting_time[i][j] + self.job[self.expert_process_job[i][j]][1])
                            delete_index.append(j)
                if len(delete_index) > 0:
                    if len(delete_index) > 1:
                        delete_index.sort(reverse = True)
                    for k in delete_index:
                        del self.expert_process_job[i][k]
                        del self.expert_process_time[i][k]
            ## calculate total time consumed
            self.total_time += sum(self.job_waiting_time[i]) + self.total_job_process_time[i].sum()
            self.expert_process_time[i] = [m + 1 for m in self.expert_process_time[i]]
        
        ## reward takes the minus of total time*0.001 and left job num
        #print(self.total_time)
        reward = 1 - self.left_job/self.job_num
        self.timeindex += 1
        
        ## update state info
        self.state = np.vstack((self.job_status,self.job_distribute_time))
        self.state = self.state.reshape(self.state.shape[0],self.state.shape[1],1)
        
        if self.left_job == 0:
            self.done = True
        #print(self.expert_status)
        #print(self.expert_process_job)
        #print(self.done_job)
        return self.state, reward, self.done, self.done_job, self.done_expert, self.job_start_time

    def update(self,delete_list):
        if len(delete_list) != 0:
            for i in delete_list:
                if i in self.job_index:
                    self.job_index.remove(i)
        else:
            pass