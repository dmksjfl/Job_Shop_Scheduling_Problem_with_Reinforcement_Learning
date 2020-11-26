class Expert(object):
    def __init__(self, expert_id, matrix_row):
        '''
        expert_id 专家的编号
        ability   能力 dict question_id -> question_time_cost
        
        '''
        # 专家的静态特性
        self.expert_id = expert_id
        self.ability = {}
        for question_id, question_time_cost in enumerate(matrix_row):
            self.ability[question_id] = question_time_cost
        self.reset()
        
    def reset(self):
        # 专家的动态特性（每次训练需要reset）
        self.working_jobs = []
        self.total_work_time = 0
    
    def can_do_question_ids(self):
        return set([q for q in self.ability.keys()])
    
    def __str__(self):
        return ','.join([str(self.expert_id), str(self.total_work_time)])
    
    def L(self):
        '''
        专家的负荷
        '''
        l = self.total_work_time / (60 * 24)
        return l
    
    def set_working_job(self, task):
        # 满足可以添加任务
        assert len(self.working_jobs) <= 2
        # 满足该任务可以处理
        assert self.ability[task.question_id] != 999999
        # 添加任务
        self.working_jobs.append(task)
    
    def remove_working_job(self, task):
        # 当前存在任务
        assert len(self.working_jobs) > 0
        #
        self.working_jobs.remove(task)
    
    
    def update_work_time(self):
        pass
    
    # call at each time update
    def update(self):
        if len(self.working_jobs) > 0:
            self.total_work_time += 1
        
    
if __name__ == "__main__":
    pass
    