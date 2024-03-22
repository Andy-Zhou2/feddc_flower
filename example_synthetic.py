from utils_general import *
from utils_dataset import DatasetObject, DatasetSynthetic
from utils_models import client_model
from utils_general import get_mdl_params, parameters_to_weights, weights_to_parameters
from client import client_fn
from functools import partial

import torch
import numpy as np
import flwr as fl
from client import FeddcClient
import os, shutil
from typing import Any
from strategy_feddc import FedDC
from utils_methods_FedDC import train_FedDC
# Dataset initialization
data_path = 'Folder/' # The folder to save Data & Model
###
alpha = 0.0
beta = 0.0
iid_sol = True
iid_data = True
name_prefix = "syn_alpha-" + str(alpha) + "_beta-" + str(beta)

n_dim = 30
n_clnt = 20
n_cls = 5
avg_data = 200

data_obj = DatasetSynthetic(alpha=alpha, beta=beta, iid_sol=iid_sol, iid_data=iid_data, n_dim=n_dim, n_clnt=n_clnt,
                            n_cls=n_cls, avg_data=avg_data, data_path=data_path, name_prefix=name_prefix)

###
# Common hyperparameters
com_amount = 300
save_period = 100
weight_decay = 1e-5
batch_size = 10
act_prob = 0.15
model_name = 'Linear'  # Model type
suffix = model_name
lr_decay_per_round = 1

def model_func():
    return client_model(model_name, [n_dim, n_cls])


init_model = model_func()


# Initalise the model for all methods
with torch.no_grad():
    init_model.fc.weight = torch.nn.parameter.Parameter(torch.zeros(n_cls, n_dim))
    init_model.fc.bias = torch.nn.parameter.Parameter(torch.zeros(n_cls))
init_weights = get_mdl_params(init_model)
n_par = init_weights.size


print('FedDC')

epoch = 10
alpha_coef = 0.005
learning_rate = 0.1

n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_clnt
n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)

n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
learning_rate = 0.1
print_per = 5


feddc_config = {
    'alpha': alpha_coef,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'epoch': epoch,
    'print_per': print_per,
    'weight_decay': weight_decay,
    'dataset_name': data_obj.dataset,
    'lr_decay_per_round': lr_decay_per_round,
}

strategy = FedDC(
    data_obj=data_obj,
    model_func=model_func,
    config=feddc_config,
    fraction_fit=act_prob,
    fraction_evaluate=0,
    initial_parameters=weights_to_parameters(get_mdl_params(init_model, n_par)),
    n_par=n_par,
    # client_sim_path=client_sim_path,
)

fl.simulation.start_simulation(
    client_fn=lambda x: client_fn(data_obj=data_obj, model_name=model_name, n_par=n_par, cid=x),
    num_clients=n_clnt,
    config=fl.server.ServerConfig(num_rounds=com_amount),
    strategy=strategy,
    client_resources={"num_cpus": 16, "num_gpus": 1},
)
