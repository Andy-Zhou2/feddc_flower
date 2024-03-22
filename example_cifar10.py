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
from utils_methods_FedDC import train_FedDC
# Dataset initialization
data_path = 'Folder/' # The folder to save Data & Model

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in data_path + "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
#      - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, name, crop_amount = 2000)
#########


n_client = 100
# Generate IID or Dirichlet distribution
# IID
data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)
# unbalanced
#data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0.3, data_path=data_path)

# Dirichlet (0.6)
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.6, data_path=data_path)
# Dirichlet (0.3)
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, seed=20, unbalanced_sgm=0, rule='Drichlet', rule_arg=0.3, data_path=data_path)

model_name = 'cifar10_LeNet' # Model type

###
# Common hyperparameters

com_amount = 1000
save_period = 200
weight_decay = 1e-3
batch_size = 50
#act_prob = 1
act_prob = 0.15
suffix = model_name
lr_decay_per_round = 0.998

def model_func():
    return client_model(model_name)


init_model = model_func()
init_weights = get_mdl_params(init_model)
n_par = init_weights.size

print(f'{n_client} clients, {com_amount} rounds, {model_name}, {act_prob} act_prob, dataset {data_obj.dataset}, {data_obj.rule}, {data_obj.rule_arg}')


# Initalise the model for all methods with a random seed or load it from a saved initial model
torch.manual_seed(37)

####

print('FedDC')

epoch = 5
alpha_coef = 1e-2
learning_rate = 0.1
print_per = epoch // 2

n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)

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
    client_resources={"num_cpus": 16, "num_gpus": 1},
)
