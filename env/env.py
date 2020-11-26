import pandas as pd
import numpy as np

from expert import Expert
from task import Task


class Env(object):
    def __init__(self, path='../data/'):
        self._load_data(path)
        self.reset()
        
        
    def _load_data(self, path):
        tasks_matrix = pd.read_csv(path + 'work_order.csv',header=None).values
        expert_jobs_matrix = pd.read_csv(path + 'process_time_matrix.csv',header=None).drop([0]).values
        
        # index start from zero!
        self.tasks =[Task(*row) for row in tasks_matrix]
        self.experts = [Expert(i, row) for i, row in enumerate(expert_jobs_matrix)]
    
    def reset(self):
        # 待处理的栈
        self.time = 400
        self.task_idx = 0
        self.taskQueue = []
        self.done_tasks = []
        
        for expert in self.experts:
            expert.reset()

    def simulate(self, f):        
        
        while len(self.done_tasks) <= 8840 and self.time <= 10000: # 8840 using 100 for test
            # 添加待执行程序
            self.addTaskToQueue()
            print('-' * 30)
            print("time:{}".format(self.time))
            print('待分配个数：', self.taskQueue.__len__())
            print('已完成个数：', self.done_tasks.__len__())
            if self.done_tasks:
                print('last_done_tasks: ')
                print(self.done_tasks[-1])
            print('-' * 30)
            
            # 分配
            def alloc(task, expert):
                # 分配任务
                print('{}, {}, {}'.format(task.task_id,
                   expert.expert_id+1, self.time), file=f)
                self.taskQueue.remove(task)
                expert.assign_working_job(self, task)
            '''
            
            '''
################################################
            for task in self.taskQueue:
                for expert in self.experts:
                    
                    if (task.question_id in expert.can_do_question_ids()
                        and expert.working_jobs.__len__() < 3
                        and task.allowed_alloc_num > 0):
                            
                            
                            alloc(task, expert)
                            break
#################################################
            self.update()
        return
    
    
    def update(self):
        # 时间步更新
        self.time += 1
        for export in self.experts:
            export.update(self)
        
        # check finish
    
    def addTaskToQueue(self):
        '''
        添加待分配的任务
        '''
        print(self.tasks.__len__())
        while (self.task_idx < 8840
         and self.tasks[self.task_idx].begin_time <= self.time):
            task = self.tasks[self.task_idx]
            # 设置项目开始时间
            assert task.begin_time == self.time
            self.taskQueue.append(task)
            self.task_idx += 1
    


if __name__ == "__main__":
    e = Env()
    with open('output.csv', 'w') as f:
        e.simulate(f)
    
#    print(len(e.tasks), e.tasks[0]) # 8840 1,481
#    print(len(e.experts), e.experts[0]) # 8840 1,481
    