import torch
import uuid

"""
   This repository contains the implementation of Fed-CBT algorithm, its training and testing procedure.
   After executing simulate_data.py and obtaining the simulated dataset, one needs to execute demo.py.
   PS: If one wants to set hyperparameters, they should go to config.py before executing demo.py
    Inputs:
        model of any device:        Deep Graph Normalizer (DGN). It can be found at model.py
        whole dataset: (N_Subjects, N_Nodes, N_Nodes, N_views)
            every fold's dataset: (N_Subjects//n_fold, N_Nodes, N_Nodes, N_views)
                every fold's test dataset:  ((N_Subjects//n_fold), N_Nodes, N_Nodes, N_views)
                every fold's train dataset: (N_Subjects-(N_Subjects//n_fold), N_Nodes, N_Nodes, N_views)
                    every device's train dataset in every fold: (N_Subjects-(N_Subjects//n_fold), N_Nodes, N_Nodes, N_views)

    Outputs:
        for every fold
            for every client
                loss plot of every device individually in a png form
                loss plot of every device after the device is updated with global model in a png form
                saved model in a torch model form
                all connectomic brain templates (cbt) in a numpy form
                fused cbt in a numpy form
            final loss of all clients in a txt form
    In order to evaluate Fed-CBT 3-fold cross-validation strategy is used.
    ---------------------------------------------------------------------
    Hızır Can Bayram
"""

# HYPERPARAMETERS #
Dataset_name = 'ASD_LH'#ASD_LH NC_LH NC_RH
# Path_root = '/data/userdisk2/wqy/TMI-FedMeta/inputs'
Path_root = '/data/userdisk1/fy/TMI/inputs'

Cluster_name ="GaussianMixture"#SIMLR"#K-means AP GaussianMixture SIMLR_PY
Setup_name = 'federation-scaffold'# for baseline and ablation
## federation-fednyn for FedNyn
#federation-fednova for FedNova
#federation-scaffold for scaffold
#federation-moon-local for MOOn
#federation-prox-local for FedProx
#no-federation-rdgn for trained DGN and RDGN
#federation
#no-federation
#no-federation-simulate
#federation-simulate
#meta-federation-simulate
#meta-federation-simulate-local
random_num = 1# for unsupervised method, default is 10
iter=10
simulate_num = 10
Start_simulate_num = 300#/0
test_num=1
bias=0.4
SEED=11
convergence_limit=8
con_begin=0.04
early_stop_limit = 20  # tells if how many rounds a model doesn't improve, it's stopped to train
N_max_epochs = 10  # 500
max_epochs_MLP = 20 #600
MLP_round = 20
n_folds =4 # cross validation fold number
number_of_samples = 3# how many c we want to use for federated learning
numEpoch = 2# how many round we want to train in an epoch
random_size = 0.1
lr = 0.0005
data_dir = "./ASD LH"

N_views = 6
N_Nodes = 35
early_stop = True
model_name = "DGN_TEST"
CONV1 = 36
CONV2 = 24
CONV3 = 5
# HYPERPARAMETERS #


N_Subjects = None
if 'ASD' in Dataset_name:
    N_Subjects = 155
else:
    N_Subjects = 186

if 'MGN' in Setup_name:
    istp = True
else:
    istp = False

temporal_weighting = False

C_sgd = None  # 1/3 # 0.91
if 'no' in Setup_name:
    C_sgd = 1
else:
    C_sgd = 0.9

if 'local' in Setup_name:
    isLocal = True
    #C_sgd = 1
else:
    isLocal = False

isFederated = None
if 'no' in Setup_name:
    isFederated = False
else:
    isFederated = True

average_all = None
if 'no' in Setup_name:
    average_all = None
else:
    average_all = False

isSimulate = None
if 'simulate' in Setup_name:
    isSimulate = True
else:
    isSimulate = False

#means supervised
if 'meta' in Setup_name:
    isMeta = True
    DGN_local_epotch_for_MLP = 5
else:
    isMeta = False
    DGN_local_epotch_for_MLP = 1


if "weight" in Setup_name:
    istemp = True
else:
    istemp = False
if 'temp' in Setup_name:
    temporal_weighting = True
    average_all = True
else:
    average_all = None
    temporal_weighting = None

Path_input = 'inputs/' + Dataset_name + '/'
Path_output = 'output/' + Dataset_name + '/' + Setup_name + '/'
Path_output_CBT = 'output/' + Dataset_name + '/' + "no-federation-rdgn" + '/'  # 'output/' + Dataset_name + '/' + "no-federation" + '/'
Path_output_rdgn = 'output/' + Dataset_name + '/' + "no-federation-rdgn" + '/'
TEMP= "./temp_mlp/"+ Dataset_name+Setup_name  + '/'
TEMP1= "./temp_mlp1/"+ Dataset_name+Setup_name  + '/'
TEMP_FOLDER = "./temp"
TEMP_FOLDER1 = "./temp1"
TEMP_FOLDER2 = "./temp2"
TEMP_FOLDER3 = "./temp3"
TEMP_FOLDER4 = "./temp4"
TEMP_FOLDER5 = "./temp5"
TEMP_FOLDER6 = "./temp6"
TEMP_FOLDER7 = "./temp7"
TEMP_FOLDER8 = "./temp8"
TEMP_FOLDER9 = "./temp9"

T1 = "./ASDLH1"
T2= "./ASDLH2"
T3 = "./ASDLH1-1"
T4= "./ASDLH2-1"

model_id = str(uuid.uuid4())

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

MODEL_PARAMS = {
    "N_ROIs": N_Nodes,
    "learning_rate": lr,
    "lambda_kl": 10,
    "n_attr": N_views,
    "Linear1": {"in": N_views, "out": CONV1},
    "conv1": {"in": 1, "out": CONV1},

    "Linear2": {"in": N_views, "out": CONV1 * CONV2},
    "conv2": {"in": CONV1, "out": CONV2},

    "Linear3": {"in": N_views, "out": CONV2 * CONV3},
    "conv3": {"in": CONV2, "out": CONV3}
}



