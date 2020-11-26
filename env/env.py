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
        self.time = 0
        self.task_idx = 0
        self.taskQueue = []
        self.done_tasks = []
        
        for expert in self.experts:
            expert.reset()

    def simulate(self):        
        
        while len(self.done_tasks) <= 100: # 8840 using 100 for test
            # 添加待执行程序
            self.addTaskToQueue()
            print("time:{}".format(self.time))
            
            
            yield 123
            print(action)
#            yield action
            
            # dosomething
            
            # showStatus
            self.update()
        return
    
    
    def update(self):
        self.time += 1
        for export in self.experts:
            export.update()
    
    def addTaskToQueue(self):
        '''
        添加待分配的任务
        '''
        while self.tasks[self.task_idx].begin_time <= self.time:
            self.taskQueue.append(self.tasks[self.task_idx])
            self.task_idx += 1
    


#class Agent(object):
#    def __init__(self, env: Env):
#        self.env = env
#        self.state
#        
#    def choose_action(self):
#        return np.random.choice(1, 4)


if __name__ == "__main__":
    e = Env()
    
    e.simulate()
    
    print(len(e.tasks), e.tasks[0])
    print(len(e.experts), e.experts[0])
    