## EKF_visual ##
import torch
import numpy as np
from torch import autograd

from util_tloc import ext_feat_and_label

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

class ExtendedKalmanFilter:
    def __init__(self, SystemModel,dataset_name ,prior_flag, encoder_type, model_encoder_trained):
        self.prior_flag = prior_flag
        self.encoder_type = encoder_type
        self.dataset_name = dataset_name
        if self.encoder_type == "torchNN":
            self.encoder = model_encoder_trained.double()
        else:
            self.encoder = model_encoder_trained

        self.f_function = SystemModel.f_function
        self.m = SystemModel.m
        self.givenQ = SystemModel.givenQ

        self.H = SystemModel.H
        self.n = SystemModel.n
        self.givenR = SystemModel.givenR

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Pre allocate KG array
        #self.KG_array = torch.zeros((self.T_test, self.m, self.m))

        # Full knowledge about the model or partial? (Should be made more elegant)
        # if (mode == 'full'):
        #     self.fString = 'ModAcc'
        #     self.hString = 'ObsAcc'
        # elif (mode == 'partial'):
        #     self.fString = 'ModInacc'
        #     self.hString = 'ObsInacc'

    # Predict
    def Predict(self):
        ####### Predict the 1-st moment of x [F(x)]
        if self.dataset_name == "tloc":
            self.F_matrix = self.f_function(self.x_t_est)
            self.x_t_given_prev = torch.matmul(self.F_matrix, self.x_t_est.float())
        # Predict the 1-st moment of y  [H*F*x]
        self.y_t_given_prev = torch.matmul(self.H.double(), self.x_t_given_prev.double(), )
        # Compute the Jacobians
        self.UpdateJacobians(self.F_matrix, self.H)

        ####### Predict the 2-nd moment of x  cov(x)=[F*Cov(x)*F_T+Q]
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior.double())
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.givenQ
        self.m2y = torch.matmul(self.H, self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.givenR

        # plt.imshow(self.y_t_given_prev)
        # plt.show()
        # plt.imshow(tmp.reshape(28,28))
        # plt.show()
        # plt.imshow(self.H.reshape(28,28))
        # plt.show()
        # Predict the 2-nd moment of y  cov(x)=[H*Cov(x)*H_T+R]

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG.double().cpu(),torch.inverse(self.m2y.cpu().detach()+ 1e-8 * np.eye(self.m)))
        # Save KalmanGain
        #self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.y_t_given_prev

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.x_t_est = self.x_t_given_prev + torch.matmul(self.KG.to(dev),self.dy.double().to(dev))
        self.x_t_est = self.x_t_est.float()
        #self.x_t_est = self.x_t_est.transpose(0, 1)
        # Compute the 2-nd posterior moment  (???)
        self.m2x_posterior = torch.matmul(self.KG.to(dev), self.m2y.double().to(dev))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.m2x_posterior.to(dev),self.KG.to(dev).transpose(0,1))

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()
        return self.x_t_est, self.m2x_posterior

    def InitSequence_EKF(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0


    def UpdateJacobians(self, F, H):
        self.F = F.double()
        self.F_T = torch.transpose(F, 0, 1).double()
        self.H = H.double()
        self.H_T = torch.transpose(H, 0, 1).double()

    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, obs, d, obs_target, T = None):
        # sequence length
        if self.dataset_name == "tloc":
            vobs = obs[["vLon", "vLat"]]
            vobs_first_index = vobs.index[0]
            obs, obs_target = ext_feat_and_label(obs)
        if not T:
            T = len(obs)

        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[T, self.m]) #space for estimation of state sequence
        self.encoder_output = torch.empty(size=[T, d]) #space for encoder estimation
        self.sigma = torch.empty(size=[T, self.m, self.m]) #covariance state noise

        # Pre allocate KG array
        #self.KG_array = torch.zeros((T, self.m, self.m))   # space for KG of each time step
        self.i = 0  # Index for KG_array allocation

        self.x_t_est = self.m1x_0
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        # prediction of raf encoder
        if self.encoder_type == "RaF":
            self.encoder_output = self.encoder.predict(obs)

        for t in range(0, T):
            if self.encoder_type == "torchNN":
                self.encoder.eval()
                # prediction of nn
                obsevation_t = obs[t]
                if self.dataset_name == "tloc":
                    if self.prior_flag:
                        encoder_output_t = torch.squeeze(self.encoder(obsevation_t.double(),self.x_t_est[0,:].unsqueeze(0).double()))
                    else:
                        encoder_output_t = torch.squeeze(self.encoder(obsevation_t))
                    self.encoder_output[t,:] = encoder_output_t
            
            y = torch.from_numpy(self.encoder_output[t,:])
            y = y.reshape((-1,1))
            if self.dataset_name == "tloc":
                vLon = torch.tensor([vobs["vLon"].loc[vobs_first_index+t]]).unsqueeze(0)
                vLat = torch.tensor([vobs["vLat"].loc[vobs_first_index+t]]).unsqueeze(0)
                y = torch.cat((y, vLon, vLat), dim=0)
                self.x_t_est[2] = vLon
                self.x_t_est[3] = vLat
            xt, sigmat = self.Update(y)
            self.sigma[t,:,:] = torch.squeeze(sigmat)
            self.x[t,:] = torch.squeeze(xt)
