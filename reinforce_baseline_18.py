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
        self.F_lim = 30
        self.dF_max = 2
        
        self.xs = []
        self.dxs = []
        self.thetas = []
        self.dthetas = []
        self.Fs = []
        
        self.Z = init_state[0:4]
        self.F_prev = init_state[4]
        
        self.last_deltas_theta = []
        self.last_deltas_x = []
    
    def step(self, F, t, x_star, pos_done_count, ang_done_count):            

        '''
        dF = F - self.F_prev
        if dF > self.dF_max:
            dF = self.dF_max
        elif dF < -self.dF_max:
            dF = -self.dF_max
        
        F = self.F_prev + dF
        '''

        Z_new = rk4(f, self.h, self.Z, F)

        self.thetas.append(Z_new[0])
        self.dthetas.append(Z_new[1])
        self.xs.append(Z_new[2])
        self.dxs.append(Z_new[3])
        self.Fs.append(self.F_prev)

        self.Z = Z_new
        self.F_prev = F   

        '''
        ft = softplus(torch.tensor(t-200).float()).numpy()

        x_penalty = (10 - Z_new[2])**2
            

        # reward = - (x_penalty + ft*theta_penalty/50) + done_bonus
        reward = -x_penalty + done_bonus
        '''


        if len(self.last_deltas_x) < 6:
            self.last_deltas_x.append(abs(Z_new[2]-x_star))
        else:
            self.last_deltas_x.pop(0)
            self.last_deltas_x.append(abs(Z_new[2]-x_star))

        
        if len(self.last_deltas_theta) < 20:
            self.last_deltas_theta.append(abs(Z_new[0]))
        else:
            self.last_deltas_theta.pop(0)
            self.last_deltas_theta.append(abs(Z_new[0]))
 

        done = (np.all(np.array(self.last_deltas_x) <= x_buffer) and np.all(np.array(self.last_deltas_theta) <=theta_buffer))

        #pos_done = np.all(np.array(self.last_deltas_x) <= x_buffer)


        pos_done = abs(Z_new[2]-x_star) <= x_buffer

        if pos_done:
            pos_done_count+=10
        else:
            pos_done_count = 0

        '''
        ang_done = np.all(np.array(self.last_deltas_theta) <=theta_buffer)
        if ang_done:
            ang_done_count +=5
        else:
            ang_done_count = 0

        both_done = pos_done and ang_done

        if both_done:
            done_bonus = pos_done_count + ang_done_count
        else:
            done_bonus = pos_done_count

        if done:
            done_count+=10
        else:
            done_count = 0

        '''
        done_bonus = pos_done_count

        dist = x_star - Z_new[2]
        dist_pen = -dist**2

        theta_pen = 0
        #-softplus(torch.tensor((Z_new[0]*100)**2 - (theta_buffer*100)**2).float()).numpy()

        #if dist <= 0:
        #dist_pen = -dist**2
        #else:
        #    dist_pen = -300*dist

        reward = dist_pen + theta_pen + done_bonus
               
        
        state = np.append(Z_new, F)
        #state = Z_new
        return state, reward, done, pos_done_count, ang_done_count
        #done_count
    
    def plot(self):
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
        self.affine2_p = nn.Linear(64,41)
        #self.affine3_p = nn.Linear(41, 41)
        
        self.affine1_v = nn.Linear(3, 64)
        # self.affine2_v = nn.Linear(64,64)
        self.affine3_v = nn.Linear(64, 1)

        self.saved_log_probs = []
        self.rewards = []
        self.saved_state_vals = []

    def forward(self, inputs):
        x1_p = F.relu(self.affine1_p(inputs))
        #x1_p = self.affine1_p(inputs)

        x2_p = self.affine2_p(x1_p)
        # x2_p = F.relu(self.affine2_p(x1_p) + x1_p)

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
        self.F_limit = 2

    def step(self, pos, cur_F, target):
        e = target - pos

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

LOAD_PATH = 'rebase16_18_a.pth'
SAVE_PATH = 'crane_sat_antiwind'+ run_num + '.pth'

policy = Policy().cuda()
#policy.load_state_dict(torch.load(LOAD_PATH))

optimizer = optim.Adam(policy.parameters(), lr=0.001, betas=(0.90,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
eps = np.finfo(np.float32).eps.item()

run_time = 99
task_time = run_time
avg_reward = 0

p_loss_graph = []
v_loss_graph = []
task_time_graph = []
reward_graph = []
batch_size = 5

print("Training")

epochs = 200000

for i_episode in range(epochs):
    # if i_episode < epochs / 2:
    #init_state = np.random.normal([0,0,0,0,0], [.1,.025, 0.5, .15, 0.5])
    '''
    else:
    '''
    init_state = np.array([0,0,0,0,0])

    crane = crane_env(init_state)
    pid_block = pid()
    int_mod_pos = integrator()
    int_mod_ang = integrator()

    state = np.append(init_state, 0)

    last_reward = 0
    ep_time = run_time
    done_count = 0
    pos_done_count = 0
    ang_done_count = 0

    for t in range(run_time+1):  # Don't infinite loop while learning
        
        # 10
        x_star = 10
        # pid_block.step(state)
        
        ang = state[0]
        pos = state[2]
        cur_F = state[4]

        pid_state_pos = int_mod_pos.step(pos, cur_F, 10)
        pid_state_ang = int_mod_ang.step(ang, cur_F, 0)

        # pid_state = np.append(pid_state_pos, pid_state_ang)

        action = select_action(policy, pid_state_pos)
        dF = (action - 20)/ 10
        
        # new_F = pid_out[0] + res_out
        
        #env_var, reward, done, done_count = crane.step(dF, t+1, x_star, done_count)
        env_var, reward, done, pos_done_count, ang_done_count = crane.step(dF, t+1, x_star, pos_done_count, ang_done_count)

        state = np.append(env_var, t+1)
        
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
np.save('crane_sat_antiwind_time' + run_num + '.npy', task_time_graph)