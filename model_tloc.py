import torch
import numpy as np

m = 4
n = 32
H = torch.eye(m)
# H = torch.from_numpy(np.array([1,0])).unsqueeze(0).float()
F = torch.tensor([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]).float()
m1x_0 = torch.from_numpy(np.zeros((4,1)))
m2x_0 = 0 * 0 * torch.eye(m)
T = 20
T_test = 50
d = 2
real_q2 = 0.001
prior_r2 = 8

# return next state
# def f_function(prev_state):
#     next_state = prev_state
#     next_state[0] = prev_state[0] + prev_state[2]
#     next_state[1] = prev_state[1] + prev_state[3]

#     return next_state

#return F matrix
def f_function(prev_state):
    return F

# not used
def h_function(state):
    return 1