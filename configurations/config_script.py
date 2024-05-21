import numpy as np
import yaml

def yaml_configuration(path):
    with open(path, "r") as stream:
      try:
        param_dict = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
        print(exc)
    return param_dict

############ parser ######################################
param_dict = yaml_configuration('./configurations/config_file.yaml')
########### Data #########################################
dataset_name = param_dict.get("dataset_name")
sinerio = param_dict.get("sinerio")
data_gen_flag = param_dict.get("data_gen_flag")
real_r2 = param_dict.get("real_r2")
#real_q2 = param_dict.get("real_q2")
## for data generation ###
if dataset_name == "tloc":
    if sinerio == "Baseline" or sinerio == "Decimation":
        T = 400
        N_E = 1000
        N_CV = 100
        T_test = 400
        N_T = 100

########### Directories ##################################
folder_tloc_data = param_dict.get("folder_tloc_data")
folder_KNetLatent_models = param_dict.get("folder_KNetLatent_model")+"/{}".format(dataset_name)+"/"
folder_simulations = param_dict.get("folder_simulations")+"/{}".format(dataset_name)+"/"
folder_encoder_model = param_dict.get("folder_encoder_model")+"/{}".format(dataset_name)+"/"
########### Architecture #################################
load_KNetLatent_trained = param_dict.get("load_KNetLatent_trained")
flag_Train = param_dict.get("flag_Train")
fix_encoder_flag = param_dict.get("fix_encoder_flag")
prior_flag = param_dict.get("prior_flag")
encoder_type = param_dict.get("encoder_type")
warm_start_flag = param_dict.get("warm_start_flag")
############ Hyper Parameters ##################
lr_kalman = param_dict.get("lr_kalman")
wd_kalman = param_dict.get("wd_kalman")
batch_size = param_dict.get("batch_size")
epoches = param_dict.get("epoches")
########### EKF ##########################################
Evaluate_EKF_flag = param_dict.get("Evaluate_EKF_flag")
EKF_with_encoder_flag = param_dict.get("EKF_with_encoder_flag")
if dataset_name == "tloc":
    dict = {0.9: 0, 0.4: 1, 0.1: 2, 0.01: 3}
    if sinerio == "Baseline":
        if prior_flag:
            db = [-4,-5,-6.5,-8.36]
            given_r = [np.power(10, x / 10) for x in db]
            given_q = [0.0001,0.001,0.01,0.1,0.3,1]
                #[0.001,0.0007,0.0007,0.001]
        else:
            db = [0,-1,-3,-5.7]
            given_r = [np.power(10, x / 10) for x in db]
            given_q = [0.3,0.5,0.7,0.9]