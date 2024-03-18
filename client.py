import flwr as fl
from flwr.common import Metrics
from utils_dataset import DatasetObject
from utils_models import client_model
from utils_general import get_mdl_params, set_client_from_params, train_model_FedDC
import os
import numpy as np
from flwr.common.typing import NDArrays, Scalar
from typing import Any
from utils_general import get_acc_loss
import torch


class FeddcClient(fl.client.NumPyClient):
    def __init__(self, cid, net, client_x, client_y, n_par=None, data_path='./client_sim/'):
        print('client init')
        self.cid = cid
        self.net = net
        self.client_x = client_x
        self.client_y = client_y
        self.n_par = n_par if n_par is not None else get_mdl_params([self.net])[0]
        self.data_path = data_path
        with open(os.path.join(data_path, 'client_' + str(cid) + '_local_update_last.npy'), 'rb') as f:
            self.local_update_last = np.load(f)
        with open(os.path.join(data_path, 'client_' + str(cid) + '_params_drift.npy'), 'rb') as f:
            self.params_drift = np.load(f)


    def get_parameters(self, config):
        return get_mdl_params([self.net], self.n_par)[0]

    def fit(self, parameters: NDArrays, config: dict[str, Any]) -> tuple[NDArrays, int, dict[str, Any]]:
        set_client_from_params(self.net, parameters)

        round_num = config['round_num']
        print('round_num', round_num)
        new_model = train_model_FedDC(
            model=self.net,
            model_func=config['model_func'],
            alpha=config['alpha'],
            local_update_last=self.local_update_last,
            global_update_last=config['global_update_last'],
            global_model_param=torch.tensor(parameters),
            hist_i=self.params_drift,
            trn_x=self.client_x,
            trn_y=self.client_y,
            learning_rate=config['learning_rate'] * (config['lr_decay_per_round'] ** (round_num - 1)),
            batch_size=config['batch_size'],
            epoch=config['epoch'],
            print_per=config['print_per'],
            weight_decay=config['weight_decay'],
            dataset_name=config['dataset_name'],
            sch_step=config['sch_step'],
            sch_gamma=config['sch_gamma'],
        )
        print('after client feddc train')

        new_model_params = get_mdl_params([new_model])

        local_update_last = new_model_params[0] - parameters[0]
        self.local_update_last = local_update_last
        self.params_drift[0] += local_update_last[0]

        with open(os.path.join(self.data_path, 'client_' + str(self.cid) + '_local_update_last.npy'), 'wb') as f:
            np.save(f, local_update_last)
        with open(os.path.join(self.data_path, 'client_' + str(self.cid) + '_params_drift.npy'), 'wb') as f:
            np.save(f, self.params_drift)

        return_param = [get_mdl_params([new_model])[0] + self.params_drift[0]]
        print('return_param[0]:', return_param[0].shape)
        print('num samples: ', len(self.client_y))
        return return_param, len(self.client_y), dict()

    def evaluate(self, parameters, config):
        raise NotImplementedError
        # set_client_from_params(self.net, parameters)
        # loss, accuracy = test(self.net, self.valloader)
        # return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(data_obj: DatasetObject, cid: str, model_name: str, n_par=None) -> fl.client.Client:
    """Load data and model for a specific client."""
    # Load model
    model_func = lambda: client_model(model_name)

    client_x = data_obj.clnt_x[int(cid)]
    client_y = data_obj.clnt_y[int(cid)]

    return FeddcClient(cid, model_func(), client_x, client_y, n_par).to_client()


def evaluate_fn(data_obj: DatasetObject, model_name: str, server_round: int, parameters: NDArrays, fed_eval_config: dict[str, Any]):
    model_func = lambda: client_model(model_name)
    cur_cld_model = set_client_from_params(model_func(), parameters)
    loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, cur_cld_model, data_obj.dataset, 0)
    return loss_tst, {'accuracy': acc_tst}

