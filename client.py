import flwr as fl
from utils_dataset import DatasetObject
from utils_models import client_model
from utils_general import get_mdl_params, set_client_from_params, train_model_FedDC
import numpy as np
from flwr.common.typing import NDArrays, Scalar
from typing import Any
from utils_general import get_acc_loss, basic_array_deserialisation, ndarray_to_array, device
import torch
from collections import OrderedDict
from flwr.common import ParametersRecord


class FeddcClient(fl.client.NumPyClient):
    def __init__(self, cid, net, client_x, client_y, model_func, n_par=None, data_path='./client_sim/'):
        self.cid = cid
        self.net = net
        self.client_x = client_x
        self.client_y = client_y
        self.n_par = n_par if n_par is not None else get_mdl_params(self.net)
        self.data_path = data_path
        self.model_func = model_func


    def fit(self, parameters: NDArrays, config: dict[str, Any]) -> tuple[NDArrays, int, dict[str, Any]]:
        print(f'--------training client {self.cid}-------------')
        parameters = parameters[0]
        set_client_from_params(self.net, parameters)
        self.net.to(device)
        self.net.train()
        for params in self.net.parameters():
            params.requires_grad = True
        
        # retrieve state_gradient_diffs and param_drifts from state
        state = self.context.state
        saved_state = state.parameters_records.get('saved_states', ParametersRecord(None))
        if not saved_state:  # None
            state_gradient_diff = np.zeros(self.n_par)
            params_drift = np.zeros(self.n_par)
        else:
            state_gradient_diff = basic_array_deserialisation(saved_state['state_gradient_diff']).copy()
            params_drift = basic_array_deserialisation(saved_state['params_drift']).copy()


        weight = len(self.client_y) / config['mean_sample_num']
        global_update_last = np.frombuffer(config['global_update_last'], dtype=np.float64)  # protocol: use float64
        global_update_last = global_update_last / weight
        alpha = config['alpha'] / weight

        round_num = config['round_num']
        new_model = train_model_FedDC(
            model=self.net,
            model_func=self.model_func,
            alpha=alpha,
            local_update_last=state_gradient_diff,
            global_update_last=global_update_last,
            global_model_param=torch.tensor(parameters).to(device),
            hist_i=torch.tensor(params_drift).to(device),
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

        new_model_params = get_mdl_params(new_model)
        delta_param_curr = new_model_params - parameters
        params_drift += delta_param_curr
        n_mini_batch = np.ceil(config['mean_sample_num'] / config['batch_size']) * config['epoch']
        beta = 1 / n_mini_batch / config['learning_rate']

        state_g = state_gradient_diff - global_update_last + beta * (-delta_param_curr)

        delta_g_cur = (state_g - state_gradient_diff) * weight

        saved_state = OrderedDict()
        saved_state['state_gradient_diff'] = ndarray_to_array(state_g)
        saved_state['params_drift'] = ndarray_to_array(params_drift)
        state.parameters_records['saved_states'] = ParametersRecord(saved_state)

        return [new_model_params], len(self.client_y), {
            'delta_g_cur': delta_g_cur.astype(np.float64).tobytes(),
            'drift': params_drift.astype(np.float64).tobytes(),
            'cid': self.cid,
        }

    def evaluate(self, parameters, config):
        raise NotImplementedError


def client_fn(data_obj: DatasetObject, cid: str, model_name: str, n_par=None) -> fl.client.Client:
    """Load data and model for a specific client."""
    # Load model
    model_func = lambda: client_model(model_name)

    client_x = data_obj.clnt_x[int(cid)]
    client_y = data_obj.clnt_y[int(cid)]

    return FeddcClient(cid, model_func(), client_x, client_y, model_func, n_par).to_client()


def evaluate_fn(data_obj: DatasetObject, model_name: str, server_round: int, parameters: NDArrays,
                fed_eval_config: dict[str, Any]):
    model_func = lambda: client_model(model_name)
    cur_cld_model = set_client_from_params(model_func(), parameters)
    loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, cur_cld_model, data_obj.dataset, 0)
    return loss_tst, {'accuracy': acc_tst}
