import numpy as np

import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

import sys

def calc_ep1(F, dx, dtheta, theta):
    return F - b2*dx + (dtheta**2)*l*B*math.sin(theta)

def calc_ep2(dtheta, theta):
    return b1*dtheta + g*l*B*math.sin(theta)

def calc_ep3(theta):
    return (l**2)*(B**2)*((math.cos(theta))**2)-(l**2)*B*C-D*C

def f(y, F):
    ep1 = calc_ep1(F, y[3], y[1], y[0])
    ep2 = calc_ep2(y[1], y[0])
    ep3 = calc_ep3(y[0])
    
    dZ_1 = y[1]
    dZ_2 = (l*B*math.cos(y[0])*ep1+C*ep2)/ep3
    dZ_3 = y[3]
    dZ_4 = -((l**2*B+D)*ep1 + (l*B*math.cos(y[0]))*ep2)/ep3
    dZ = np.array([dZ_1, dZ_2, dZ_3, dZ_4])
    return dZ

def rk4(f, h, y0, F):
    k1 = f(y0, F)
    k2 = f(y0 + h / 2 * k1, F)
    k3 = f(y0 + h / 2 * k2, F)
    k4 = f(y0 + h * k3, F)
    return y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

b1 = 0
b2 = 0
l = 20
m1 = .1
m2 = 5
M = 20
g = 9.81
r = 1

B = m1 + m2/2
C = m1 + m2 + M
D = 2/5*m1*(r**2)

d = 10

x_buffer = .2
theta_buffer=.05

class crane_env():
    def __init__(self, init_state):
        self.h = 0.5
        # self.F_lim = 50
        
        self.xs = []
        self.dxs = []
        self.thetas = []
        self.dthetas = []
        self.Fs = []
        
        self.Z = init_state[0:4]
        self.F_prev = 0
        # init_state[4]
        
        self.last_deltas_theta = []
        self.last_deltas_x = []
    
    def step(self, dF, t):            
        '''
        F = self.F_prev + dF
        if F > self.F_lim: F = self.F_lim
        if F < -self.F_lim: F = -self.F_lim
        '''
        F = dF

        Z_new = rk4(f, self.h, self.Z, F)

        self.thetas.append(Z_new[0])
        self.dthetas.append(Z_new[1])
        self.xs.append(Z_new[2])
        self.dxs.append(Z_new[3])
        self.Fs.append(self.F_prev)

        self.Z = Z_new
        self.F_prev = F   

        #if len(self.last_dxs) < 50:
        #    self.last_dxs.append(Z_new[3])
        #else:
        #    self.last_dxs.pop(0)
        #    self.last_dxs.append(Z_new[3])
            
        #done = abs(Z_new[2]-d) < x_buffer and np.all(np.array(self.last_dxs) <= dx_buffer)

        ft = softplus(torch.tensor(t-30).float()).numpy()

        x_penalty = (10 - Z_new[2])**2
        theta_penalty = softplus(torch.tensor((Z_new[0]*100)**2 - (theta_buffer*100)**2).float()).numpy()

        if len(self.last_deltas_x) < 6:
            self.last_deltas_x.append(abs(Z_new[2]-10))
        else:
            self.last_deltas_x.pop(0)
            self.last_deltas_x.append(abs(Z_new[2]-10))

        if len(self.last_deltas_theta) < 20:
            self.last_deltas_theta.append(abs(Z_new[0]))
        else:
            self.last_deltas_theta.pop(0)
            self.last_deltas_theta.append(abs(Z_new[0]))
            
        done = np.all(np.array(self.last_deltas_x) <= x_buffer) and np.all(np.array(self.last_deltas_theta) <= theta_buffer)

        # 0.0005*(100+self.time) * 
        
        done_bonus = 0
        if done:
            done_bonus = 100-t

        reward = - (x_penalty + ft*theta_penalty/50) + done_bonus
        # state = np.append(Z_new, F)
        state = Z_new
        return state, reward, done
    
    def plot(self, img_file):
        fig, axs = plt.subplots(1, 4, figsize=(20,5))

        axs[0].plot(range(len(self.xs)), self.xs)
        axs[0].set_title("Distance (m) vs. Time")

        axs[1].plot(range(len(self.thetas)), self.thetas, label="true")
        # plt.plot(range(len(x_stars)), x_stars, label="target")
        axs[1].set_title("Angle (rad) vs. Time")

        '''
        plt.plot(range(len(self.dxs)), self.dxs)
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.show()
        '''
        
        axs[2].plot(range(len(self.Fs)), self.Fs)
        axs[2].set_title("Force (N) vs. Time")
        
        '''
        axs[3].plot(range(len(loss_graph)), loss_graph)
        axs[3].set_title("Loss vs. Ep")
        '''
        plt.savefig(img_file)

        

gamma = .99
seed = 543
render = True
log_interval = 100
softplus = nn.Softplus()
sig = nn.Sigmoid()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1_p = nn.Linear(5, 128)
        self.affine2_p = nn.Linear(128, 41)
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, inputs):
        x = F.relu(self.affine1_p(inputs))
        action_scores = self.affine2_p(x)
        action_probs = F.softmax(action_scores, dim=1)
        
        
        return action_probs


def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state.cuda())
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    Gt = []
    
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        Gt.insert(0, R)
    
    Gt = torch.tensor(Gt)
    Gt = (Gt-Gt.mean()) / (Gt.std() + eps)

    policy_loss = []

    for log_prob, reward in zip(policy.saved_log_probs, Gt):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    
    return policy_loss.detach().cpu().numpy()

run_num = sys.argv[1]
best_reward = -np.inf
best_reward_log = -np.inf
# LOAD_PATH = 'rebase2_'+ run_num + '.pth'
SAVE_PATH = 're1_'+ run_num + '.pth'

policy = Policy().cuda()
# policy.load_state_dict(torch.load(LOAD_PATH))

optimizer = optim.Adam(policy.parameters(), lr=0.001, betas=(0.90,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
eps = np.finfo(np.float32).eps.item()

task_time = 99
avg_reward = 0

p_loss_graph = []
task_time_graph = []
reward_graph = []


print("Training")

epochs = 100000
for i_episode in range(epochs):
    # if i_episode < epochs / 2:
    init_state = np.random.normal([0,0,0,0], [.1,.025, 2, .75])
    '''
    else:
    '''
    # init_state = np.array([0,0,0,0])

    crane = crane_env(init_state)
    state = np.append(init_state, 0)
    last_reward = 0
    ep_time = 99
    for t in range(100):  # Don't infinite loop while learning
        
        action = select_action(policy, state)
        new_F = action-20
        
        env_var, reward, done, = crane.step(new_F, t+1)
        state = np.append(env_var, t+1)
        
        #if render and i_episode % log_interval == 0:
        #    env.render()
        
        policy.rewards.append(reward)
        last_reward = last_reward + reward
        
        if done and ep_time == 99:
            ep_time = t
            
    if last_reward > best_reward_log:
        best_reward_log = last_reward
        save_state_dict = policy.state_dict()

    if i_episode == 0:
        avg_reward = last_reward
    else:
        avg_reward = avg_reward * 0.99 + last_reward * 0.01
        
    task_time = task_time * 0.99 + ep_time * 0.01
    
    p_loss = finish_episode()

    p_loss_graph.append(p_loss)
    reward_graph.append(last_reward/100)
    task_time_graph.append(ep_time)

    if i_episode % log_interval == 0 and i_episode != 0:
        print('Episode {} Last length {} Average length: {:.2f} Last reward: {:.2f} Average reward: {:.2f}  \n Last p_loss: {:.2f}'.format(
            i_episode, ep_time, task_time, last_reward/100, avg_reward/100, p_loss))
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            print("Saving: Average reward" + str(best_reward/100) + " Saved reward: " + str(best_reward_log/400))
            torch.save(save_state_dict, SAVE_PATH)
            best_reward_log = -np.inf

np.save('p_loss_graph1' + run_num + '.npy', p_loss_graph)
np.save('reward_graph1' + run_num + '.npy', reward_graph)
np.save('task_time_graph1' + run_num + '.npy', task_time_graph)