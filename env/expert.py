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
        if not hasattr(self, 'can_do'):
            self.can_do = []
            for q in self.ability.keys():
                if self.ability[q] != 999999:
                    self.can_do.append(q) 
        return self.can_do
    
    def __str__(self):
        return ','.join([str(self.expert_id), str(self.total_work_time)])
    
    def L(self):
        '''
        专家的负荷
        '''
        l = self.total_work_time / (60 * 24)
        return l
    
    def assign_working_job(self, env, job):
        '''
        被指派任务
        '''
        # 满足可以添加任务 且 满足该任务可以处理
        assert len(self.working_jobs) <= 2
        assert self.ability[job.question_id] != 999999
        # 添加任务
        self.working_jobs.append(job)
        job.assigned(env.time)
    
    def remove_working_job(self, env, job):
        '''
        完成任务
        '''
        # 当前存在任务
        assert len(self.working_jobs) > 0
        #
        self.working_jobs.remove(job)
        env.done_tasks.append(job)
        job.finish(env.time)
    
    # call at each time update
    def update(self, env):
        # 记录工作时间
        if len(self.working_jobs) > 0:
            self.total_work_time += 1
        
        for job in self.working_jobs:
            assert hasattr(job, 'start_time')
            if env.time - job.start_time >= self.ability[job.question_id]:
                # 完成任务
                self.remove_working_job(env, job)
                
    
if __name__ == "__main__":
    pass
    