# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:33:07 2020

@author: lvjf

train the agent
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from job_env import job_shop_env
from RL_brain import ActorCritic
from utils import v_wrap
import csv

def train(args):
    torch.manual_seed(args.seed)

    env = job_shop_env()
    
    model = ActorCritic(env.state_dim, env.action_dim)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = v_wrap(state)
    done = True
    action_dim = env.expert

    episode_length = 0
    complete_jobs = []
    expert_complete_job = []
    complete_job_start_time = []
    update_list = []
    for episode in range(args.episode):
        
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
        
        if len(complete_jobs) != 0:
            update_list = [n for m in complete_jobs for n in m]
            env.update(update_list)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps+1):
            episode_length += 1
            

            action, log_prob, entropy, value = model.choose_action((state, (hx,cx)),action_dim)
            log_prob = log_prob.gather(1, action)[0]
            
            state, reward, done, done_job, done_expert, job_start_time = env.step(action.view(-1,).numpy())
            done = done or episode_length >= args.max_episode_length
            ## reward shaping
            reward = max(min(reward, 1), -1)
            if episode_length % 20 == 0:
                print(reward)
                #print(done_job)

            if done:
                complete_jobs.append(done_job)
                expert_complete_job.append(done_expert)
                complete_job_start_time.append(job_start_time)
                print('Complete these jobs with 100 iterations:')
                print(complete_jobs)
                print('Current episode:',episode)
                episode_length = 0
                state = env.reset()

            state = v_wrap(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            if done:
                break
        
        if len(list(set(update_list))) > 8800:
            ## write results into the csv file
            with open('submit_{}.csv'.format(len(list(set(update_list)))),'w') as f:
                writer = csv.writer(f)
                for i in range(len(complete_jobs)):
                    for j in range(len(complete_jobs[i])):
                        writer.writerow([complete_jobs[i][j]+1, expert_complete_job[i][j]+1, complete_job_start_time[i][j]])

        if episode == args.episode -1 or len(list(set(update_list))) == 8840:
            ## write results into the csv file
            with open('submit.csv','w') as f:
                writer = csv.writer(f)
                for i in range(len(complete_jobs)):
                    for j in range(len(complete_jobs[i])):
                        writer.writerow([complete_jobs[i][j]+1, expert_complete_job[i][j]+1, complete_job_start_time[i][j]])
            break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward(torch.ones_like(policy_loss))
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        print(policy_loss.mean() + args.value_loss_coef * value_loss)
        print('para updated')