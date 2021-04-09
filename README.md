# Job Shop Scheduling Problem with Reinforcement Learning
This is the implemention of JSSP with RL. The framework used for RL is actor critic and the dataset comes from Tianchi competition.

## Problem formulation
see [https://tianchi.aliyun.com/competition/entrance/531831/information](https://tianchi.aliyun.com/competition/entrance/531831/information) for more details

### Action design
The agent action takes from {0, 1} where the action array has the same size as the number of experts.

### Reward design
The reward takes the following formula:

r = 1 - M/N

where M is the amount of left jobs and N is the total job amount.

### State design
The state is the concat of job status and job distribution time

## How to use
- the `data` file contains the dataset for the JSSP
- the `RL_brain.py` is the reinforcement learning algorithm for the problem
- the `trainer.py` is the training function for the problem
- the `job_env.py` is the environment for the JSSP
- the `utils.py` contains useful functions that are used many times
- the `run.py` is the main file of the project

One could run the following code to train the agent:
```python
python run.py --lr=0.01 --gamma=0.9 --seed=2020 --num-steps=100 --max-episode-length=1e5
```

## Reference on better JSSP solution
A paper named `A Reinforcement Learning Environment For Job-Shop Scheduling` is published in arXiv on 4/9, 2021. One could refer to this paper for action design, state design and most importantly, reward function design. The paper also provides a JSSP standard environment and a baseline solution for the environment, which might be of great help for those who are interested in JSSP. The paper could be found [here](https://arxiv.org/pdf/2104.03760.pdf) and the corresponding code could be found [here](https://github.com/prosysscience/Job-Shop-Scheduling)
