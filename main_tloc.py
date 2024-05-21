import matplotlib.pyplot as plt
import time

from configurations.config import *
from data_initialization import initialize_data_TLoc, gen_state_data
from train.RaF_train import non_transfer_train
from tests.RaF_test import raf_test
from tests.EKF_test import EKFTest

from train.Pipeline_KF import Pipeline_KF
from Latent_KN import KalmanNetLatentNN, in_mult, out_mult

# data initialization
train_loader,val_loader,test_loader,train_input, train_target, cv_input, cv_target, test_input, test_target = initialize_data_TLoc(tloc_data_path, batch_size, prior_r2, real_r2, dev, warm_start_flag)
gen_state_data(train_input, prior_r2, real_r2, warm_start_flag)
gen_state_data(test_input, prior_r2, real_r2, warm_start_flag)
# gen_state_data(cv_input, prior_r2, real_r2, warm_start_flag)
train_domains = train_input.groupby(['RNCID_1', 'CellID_1'])
for name, domain in train_domains:
    train_domain = domain
    test_domain = test_input[(test_input['RNCID_1'] == name[0]) & (test_input['CellID_1'] == name[1])]
    break
# train_domain = train_domains.first()


# encoder training
model_encoder_trained = non_transfer_train(train_domain, model_encoder_trained)


# encoder testing
raf_test(test_domain,model_encoder_trained)


# EKF testing
if Evaluate_EKF_flag:
    r = given_r[dict.get(real_r2)]
    q = given_q[dict.get(real_r2)]
    print('given q = {} given r = {} '.format(q, r))
    sys_model.givenQ = q * q * torch.eye(m)
    sys_model.givenR = r * r * torch.eye(m)
    print("5. Evaluate Extended Kalman Filter {} {} dataset with real_q = {} and prob_r = {}".format(dataset_name,sinerio,real_q2, real_r2))
    # [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg_with_encoder, EKF_out_with_encoder, GT_test] = EKFTest(sys_model, test_input, test_target, dataset_name, sinerio,prior_flag,encoder_type, model_encoder_trained,d)
    # [median_error,error] = EKFTest(sys_model, test_input, test_target, dataset_name, sinerio,prior_flag,encoder_type, model_encoder_trained,d)
    [median_error,error] = EKFTest(sys_model, test_domain, test_target, dataset_name, sinerio,prior_flag,encoder_type, model_encoder_trained,d)


# KN training


# KN testing