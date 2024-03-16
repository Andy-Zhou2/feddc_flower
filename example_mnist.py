from utils_dataset import DatasetObject
from utils_models import client_model
from utils_general import get_mdl_params, set_client_from_params, train_model_FedDC

import torch
import numpy as np
import flwr as fl
from client import FeddcClient
import os, shutil

data_path = 'Folder/'  # The folder to save Data & Model

# Generate IID or Dirichlet distribution
# IID
n_client = 100
data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)

# Dirichlet (0.6)
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)

model_name = 'mnist_2NN'  # Model type

# Common hyperparameters
com_amount = 300
save_period = 100
weight_decay = 1e-3
batch_size = 50
act_prob = 0.15
suffix = model_name
lr_decay_per_round = 0.998

# Model function
def model_func():
    return client_model(model_name)
init_model = model_func()
n_par = get_mdl_params([init_model])[0].size

torch.manual_seed(37)

epoch = 5
alpha_coef = 0.1
learning_rate = 0.1
print_per = 5  # epoch // 2

n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)



client_sim_path = './client_sim/'
if os.path.exists(client_sim_path):
    shutil.rmtree(client_sim_path)
os.mkdir(client_sim_path)

for c in range(n_client):
    np.save(client_sim_path + 'client_' + str(c) + '_local_update_last.npy', np.zeros(n_par))
    np.save(client_sim_path + 'client_' + str(c) + '_params_drift.npy', np.zeros(n_par))

# [avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf] = train_FedDC(data_obj=data_obj, act_prob=act_prob, n_minibatch=n_minibatch,
#                                     learning_rate=learning_rate, batch_size=batch_size, epoch=epoch,
#                                     com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                     model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
#                                     sch_step=1, sch_gamma=1,save_period=save_period, suffix=suffix, trial=False,
#                                     data_path=data_path, lr_decay_per_round=lr_decay_per_round)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=FedCustom(),  # <-- pass the new strategy here
    client_resources=client_resources,
)
