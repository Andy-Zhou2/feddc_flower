from utils_dataset import DatasetObject
from utils_models import client_model
from utils_general import get_mdl_params, parameters_to_weights, weights_to_parameters
from client import client_fn, evaluate_fn
from functools import partial

import torch
import numpy as np
import flwr as fl
from client import FeddcClient
import os, shutil
from typing import Any
from strategy_feddc import FedDC


data_path = 'Folder/'  # The folder to save Data & Model

# Generate IID or Dirichlet distribution
# IID
n_client = 100
# data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)

# Dirichlet (0.6)
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)

# model_name = 'mnist_2NN'  # Model type
model_name = 'cifar10_LeNet'  # Model type

# Common hyperparameters
com_amount = 60
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
init_weights = get_mdl_params(init_model)
n_par = init_weights.size

torch.manual_seed(37)

epoch = 5
alpha_coef = 0.1
learning_rate = 0.1
print_per = epoch // 2

n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)

# client_sim_path = './client_sim/'
# if os.path.exists(client_sim_path):
#     shutil.rmtree(client_sim_path)
# os.mkdir(client_sim_path)

# for c in range(n_client):
    # np.save(os.path.join(client_sim_path, 'client_' + str(c) + '_local_update_last.npy'), np.zeros(n_par))
    # np.save(os.path.join(client_sim_path, 'client_' + str(c) + '_params_drift.npy'), np.zeros(n_par))
    # np.save(os.path.join(client_sim_path, 'client_' + str(c) + '_model_weights.npy'), init_weights)

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
    num_clients=n_client,
    config=fl.server.ServerConfig(num_rounds=com_amount),
    strategy=strategy,
    client_resources={"num_cpus": 16, "num_gpus": 0.0},
)
