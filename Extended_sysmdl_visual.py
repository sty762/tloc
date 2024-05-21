## Extended_sysmdl_visual ##
import torch
from configurations.config_script import sinerio
# from model_Lorenz import y_size, m
import numpy as np

if torch.cuda.is_available():
    cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    cuda0 = torch.device("cpu")
    #print("Running on the CPU")

class SystemModel:
    def __init__(self, f_function, given_q, real_q2, m, h_function, H, given_r, real_r, n, T, T_test, dataset_name, prior_Q=None, prior_Sigma=None, prior_S=None):
        ####################
        ### Motion Model ###
        ####################
        self.dataset_name = dataset_name
        self.f_function = f_function
        self.real_q2 = real_q2
        self.m = m
        #self.realQ = real_q * real_q * torch.eye(self.m)
        #self.givenQ = given_q * given_q * torch.eye(self.m)
        # if self.modelname == 'pendulum':
        #     self.Q = q * q * torch.tensor([[(delta_t ** 3) / 3, (delta_t ** 2) / 2],
        #                                    [(delta_t ** 2) / 2, delta_t]])
        # elif self.modelname == 'pendulum_gen':
        #     self.Q = q * q * torch.tensor([[(delta_t_gen ** 3) / 3, (delta_t_gen ** 2) / 2],
        #                                    [(delta_t_gen ** 2) / 2, delta_t_gen]])
        # else:
        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.h_function = h_function
        self.n = n
        #self.realR = real_r * real_r * torch.eye(self.n)
        #self.givenR = given_r * given_r * torch.eye(m)

        # Assign T and T_test
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.eye(self.m)
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.m)
                #self.n)
        else:
            self.prior_S = prior_S

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        #torch.squeeze(m1x_0).to(cuda0)
        self.m2x_0 = torch.squeeze(m2x_0).to(cuda0)

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    def transform_to_range(self,x):
        x_transformed = torch.empty(self.m, x.shape[1])
        x1=x[0,:]
        x1=(x1-x1.min()+1)/(x1.max()-x1.min())*25
        x2=x[1,:]
        x2=(x2-x2.min()+1)/(x2.max()-x2.min())*25
        x_transformed[0,:] = x1
        x_transformed[1,:] = x2
        x_transformed[2,:] = x[2,:]
        return x_transformed


