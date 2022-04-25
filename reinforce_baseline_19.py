import numpy as np

import math
import matplotlib.pyplot as plt
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

import sys

x_buffer = .2
theta_buffer=.05


class test_env():
    def __init__(self):
        self.F = 0
        self.Fs = []
        self.ts = []
        
        # tau * dy2/dt2 + 2*zeta*tau*dy/dt + y = Kp*u
        Kp = 1.5    # gain
        tau = 1.0   # time constant
        zeta = 0.25 # damping factor
        theta = 0.0 # no time delay
        du = 1.0    # change in u

        # (1) Transfer Function
        num = [Kp]
        den = [tau**2,2*zeta*tau,1]
        self.sys = signal.TransferFunction(num,den)
        
        self.xs = []
        self.last_deltas_x = []
        '''
        self.h = 0.5
        self.F_lim = 30
        

        self.dxs = []

        self.Z = init_state[0:2]
        
        self.last_deltas_theta = []
        self.last_deltas_x = []
        '''
    
    def step(self, dF, t, x_star, done_count):   
        new_F = self.F + dF
        self.F = new_F
        self.Fs.append(new_F)
        self.ts.append(t)
        
        tout, yout, xout = signal.lsim(self.sys, U=self.Fs, T=self.ts)
        
        if (yout.size == 1):
            last_y = yout
        else:
            last_y = yout[-1]
        
        self.xs.append(last_y)

        if len(self.last_deltas_x) < 6:
            self.last_deltas_x.append(abs(last_y-x_star))
        else:
            self.last_deltas_x.pop(0)
            self.last_deltas_x.append(abs(last_y-x_star))

        done = np.all(np.array(self.last_deltas_x) <= x_buffer)

        if done:
            done_count+=10
        else:
            done_count = 0

        reward = -(x_star - last_y)**2 + done_count
               
        return last_y, reward, done, done_count, dF
    
    
    def plot(self):
        plt.plot(range(len(self.xs)), self.xs)
        plt.show()

        

gamma = .99
seed = 543
render = True
log_interval = 100
softplus = nn.Softplus()
sig = nn.Sigmoid()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1_p = nn.Linear(3, 64)
        self.affine2_p = nn.Linear(64,21)
        #self.affine3_p = nn.Linear(41, 41)
        
        self.affine1_v = nn.Linear(3, 64)
        # self.affine2_v = nn.Linear(64,64)
        self.affine3_v = nn.Linear(64, 1)

        self.saved_log_probs = []
        self.rewards = []
        self.saved_state_vals = []

    def forward(self, inputs):
        x1_p = F.relu(self.affine1_p(inputs))
        x2_p = self.affine2_p(x1_p)

        #action_scores = self.affine3_p(x2_p)
        #action_scores = self.affine3_p(x1_p) + x1_p
        action_probs = F.softmax(x2_p, dim=1)
        
        x1_v = F.relu(self.affine1_v(inputs))
        # x2_v = F.relu(self.affine2_v(x1_v)+x1_v)

        state_value = self.affine3_v(x1_v)
        
        return action_probs, state_value


def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = policy(state.cuda())
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    policy.saved_state_vals.append(state_value)
    return action.item()


def finish_episode(run_time, batch_size):
    R = 0
    Gt = []
    j = 0

    for i in range(len(policy.rewards)-1, -1, -1):
        j+=1
        r = policy.rewards[i]
        R = r+gamma*R
        
        Gt.insert(0,R)
        
        if j % (run_time+1) == 0:
            R = 0
    
    Gt = torch.tensor(Gt)
    
    p_losses = []
    v_losses = []
    
    for log_prob, state_val, G in zip(policy.saved_log_probs, policy.saved_state_vals, Gt):
        p_losses.append(-1 * log_prob * (G - state_val.item()))
        v_losses.append(F.mse_loss(state_val, torch.tensor([[G]]).float().cuda(), reduction='mean'))
    
    optimizer.zero_grad()
    policy_loss = torch.stack(p_losses).sum()
    value_loss =  torch.stack(v_losses).sum()
    loss = (policy_loss + value_loss) / batch_size
    loss.backward()
    optimizer.step()
    
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    del policy.saved_state_vals[:]

    # print(loss.item())

    return policy_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy()

class pid():
    def __init__(self):
        self.e_prev = 0
        self.e_int = 0
        self.x_star = 10
        self.F_lim = 30
        self.max_dF = 4
        self.kp = 2
        self.kd = 5
        self.ki = -0.015
        
        self.d = 10
        self.alpha = 0.025
        self.riseTime = 80
        
        self.es = []
        self.e_ints = []
        self.x_stars = []
        
    def step(self, state):
        
        x_star = self.d/2*np.tanh(self.alpha*(state[3]-self.riseTime*2/2))+self.d/2
        
        '''
        e = x_star - state[2]

        self.e_int +=e
        e_dev = e-self.e_prev
        self.e_prev = e
        
        F = self.kp*e + self.kd*e_dev + self.ki*self.e_int
        
        F_prev = state[4]
        
        if F - F_prev > self.max_dF:
            F = F_prev + self.max_dF
        if F - F_prev < -self.max_dF:
            F = F_prev - self.max_dF
            
        if F > self.F_lim:
            F = self.F_lim
        if F < -self.F_lim:
            F = -self.F_lim

        self.es.append(e)
        self.e_ints.append(self.e_int)
        self.x_stars.append(x_star)
        '''
                    
        return x_star
        #np.array([F, e, self.e_int, e_dev])

class integrator():
    def __init__(self):
        self.int = 0
        self.prev = None
        self.F_limit = 0.5
        
    def step(self, pos, cur_F):
        e = 10- pos

        if abs(cur_F) >= self.F_limit:
            pass
        else:
            self.int +=e
        
        if not self.prev:
            dev = 0
        else:
            dev = e - self.prev
        
        self.prev = e

        return np.array([e, self.int, dev])
        
run_num = sys.argv[1]

best_reward = -np.inf
best_reward_log = -np.inf
'''
best_time = np.inf
best_time_log = np.inf
'''

LOAD_PATH = 'rebase13_13_a.pth'
SAVE_PATH = 'second_antiwind'+ run_num + '.pth'

policy = Policy().cuda()
# policy.load_state_dict(torch.load(LOAD_PATH))

optimizer = optim.Adam(policy.parameters(), lr=0.001, betas=(0.90,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
eps = np.finfo(np.float32).eps.item()

run_time = 49
task_time = run_time
avg_reward = 0

p_loss_graph = []
v_loss_graph = []
task_time_graph = []
reward_graph = []
batch_size = 5

print("Training")

epochs = 100000

for i_episode in range(epochs):
    # if i_episode < epochs / 2:
    #init_state = np.random.normal([0,0,0], [0.5, .15, 0.5])
    '''
    else:
    '''
    crane = test_env()
    pid_block = pid()
    int_mod = integrator()

    state = np.array([0,0])

    last_reward = 0
    ep_time = run_time
    done_count = 0
    cur_F = 0
    for t in range(run_time+1):  # Don't infinite loop while learning
        
        # 10
        x_star = 10
        # pid_block.step(state)
        
        pos = state[0]

        pid_state = int_mod.step(pos, cur_F)

        action = select_action(policy, pid_state)
        dF = (action - 10) / 20
        
        # new_F = pid_out[0] + res_out
        
        last_y, reward, done, done_count, cur_F = crane.step(dF, t+1, x_star, done_count)
        state = np.array([last_y, t+1])
        
        policy.rewards.append(reward)
        last_reward = last_reward + reward
        
        if done and ep_time == run_time:
            ep_time = t
            

    if last_reward > best_reward_log:
        best_reward_log = last_reward
        save_state_dict = policy.state_dict()
    '''
    if ep_time < best_time_log:
        best_time_log = ep_time
        save_state_dict = policy.state_dict()
    '''

    if i_episode == 0:
        avg_reward = last_reward
    else:
        avg_reward = avg_reward * 0.99 + last_reward * 0.01
        
    task_time = task_time * 0.99 + ep_time * 0.01

    
    task_time_graph.append(ep_time)


    if i_episode % batch_size ==0 and i_episode !=0:
        p_loss, v_loss = finish_episode(run_time, batch_size)
        
    if i_episode % log_interval == 0 and i_episode != 0:
        print('Episode {} Last length {} Average length: {:.2f} Last reward: {:.2f} Average reward: {:.2f}  \n Last p_loss: {:.2f} Last v_loss: {:.2f}'.format(
            i_episode, ep_time, task_time, last_reward/100, avg_reward/100, p_loss, v_loss))
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            print("Saving: Average reward" + str(best_reward/100) + " Saved reward: " + str(best_reward_log/400))
            torch.save(save_state_dict, SAVE_PATH)
            best_reward_log = -np.inf
        '''
        if task_time < best_time:
            best_time = task_time
            print("Saving: Average time" + str(task_time) + " Saved reward: " + str(best_time_log))
            torch.save(save_state_dict, SAVE_PATH)
            best_time_log = np.inf
        '''
'''
np.save('p_loss_graph4_' + run_num + '.npy', p_loss_graph)
np.save('v_loss_graph4_' + run_num + '.npy', v_loss_graph)
np.save('reward_graph4_' + run_num + '.npy', reward_graph)
'''
np.save('second_antiwind_' + run_num + '.npy', task_time_graph)