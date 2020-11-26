class Task(object):
    def __init__(self, task_id, time, question_id, sla_time):
        '''
         任务ID
         任务产生时间：时间为以00:00(hh:mm)为起点的分钟数
         问题分类ID
         任务最大响应时长(单位:分钟)
        '''
        self.task_id = task_id
        self.begin_time = time
        self.question_id = question_id
        self.sla_time = sla_time
        
        self.allowed_alloc_num = 5
        
    def M():
        '''
        任务的响应超时量
        '''
        assert hasattr(self, 'begin_time')
        assert hasattr(self, 'start_time')
        return max(self.start_time - self.begin_time - self.sla_time, 0) / self.sla_time
    
    def R():
        assert hasattr(self, last_stay_time)
        return self.last_stay_time / (self.finish_time - self.begin_time)
    
    
        
    def __str__(self):
        return ','.join([str(self.task_id), str(self.begin_time)])
        
if __name__ == "__main__":
    t = Task(1, 10, 105, 90)
    print(t.__dict__.keys())