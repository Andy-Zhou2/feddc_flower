from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from typing import Callable, Union
import flwr as fl
import numpy as np
from collections import defaultdict
import numbers
from utils_general import (set_client_from_params, get_acc_loss,
                           weights_to_parameters, parameters_to_weights,
                           ndarray_to_array, basic_array_deserialisation, device)
from utils_dataset import DatasetObject
import torch
import os

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


def aggregate_weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Combine results from multiple clients.

    Args:
        metrics (list[tuple[int, dict]]): collected clients metrics

    Returns
    -------
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * m for num_examples, m in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }


class FedDC(fl.server.strategy.Strategy):
    def __init__(
            self,
            model_func: Callable[[], torch.nn.Module],
            data_obj: DatasetObject,
            config: Dict[str, Scalar],
            n_par: int,
            initial_parameters: Parameters,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2
    ) -> None:
        super().__init__()
        self.model_func = model_func
        self.data_obj = data_obj
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.state_gradient_diff = np.zeros(n_par)
        self.n_par = n_par
        self.config = dict() if config is None else config

        # prepare weight_list
        # weight_list[i] is the #samples of client i / mean sample numbers per client
        weight_list = np.asarray([len(data_obj.clnt_y[i]) for i in range(data_obj.n_client)])
        self.mean_sample_num = np.mean(weight_list)
        print('mean_sample_num:', self.mean_sample_num)

        self.cent_x = np.concatenate(data_obj.clnt_x, axis=0)
        self.cent_y = np.concatenate(data_obj.clnt_y, axis=0)

        self.selected_clients_average_weight = None
        self.all_client_average_weight = None

        init_par_list = parameters_to_weights(initial_parameters)
        self.all_client_params = np.ones(data_obj.n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                                     -1)  # n_clnt X n_par
        self.all_client_drifts = np.zeros((data_obj.n_client, n_par))

    def __repr__(self) -> str:
        return "FedDC"

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_configurations = []
        for client in clients:
            config = {
                'round_num': server_round,
                'alpha': self.config['alpha'],
                'global_update_last': self.state_gradient_diff.astype(np.float64).tobytes(),
                # self.state_gradient_diff,
                'learning_rate': self.config['learning_rate'],
                'batch_size': self.config['batch_size'],
                'epoch': self.config['epoch'],
                'print_per': self.config['print_per'],
                'weight_decay': self.config['weight_decay'],
                'dataset_name': self.config['dataset_name'],
                'sch_step': 1,
                'sch_gamma': 1,
                'lr_decay_per_round': self.config['lr_decay_per_round'],
                'mean_sample_num': self.mean_sample_num,
            }
            fit_configurations.append((client, FitIns(parameters, config)))

        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [
            parameters_to_weights(fit_res.parameters) for _, fit_res in results
        ]
        drift_results = [
            np.frombuffer(fit_res.metrics['drift'], dtype=np.float64) for _, fit_res in results
        ]

        delta_g_results = [
            np.frombuffer(fit_res.metrics['delta_g_cur'], dtype=np.float64) for _, fit_res in results
        ]

        cid_results = [int(client.cid) for client, _ in results]
        delta_g_sum = np.sum(delta_g_results, axis=0)
        delta_g_cur = 1 / self.data_obj.n_client * delta_g_sum
        self.state_gradient_diff += delta_g_cur

        selected_clients_average_weight = np.mean(weights_results, axis=0)

        # update global record of client weights and drifts
        self.all_client_params[cid_results] = weights_results
        self.all_client_drifts[cid_results] = drift_results

        avg_all_client_weights = np.sum(self.all_client_params, axis=0) / self.data_obj.n_client
        avg_all_client_drifts = np.sum(self.all_client_drifts, axis=0) / self.data_obj.n_client

        cloud_model_weight = selected_clients_average_weight + avg_all_client_drifts
        all_client_average_weight = avg_all_client_weights

        parameters_aggregated = weights_to_parameters(cloud_model_weight)

        # save for evaluation
        self.selected_clients_average_weight = selected_clients_average_weight
        self.all_client_average_weight = all_client_average_weight

        return parameters_aggregated, {}

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        assert self.fraction_evaluate == 0.0, "FedDC does not support client-side evaluation"
        return []

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        results = dict()
        cur_cld_model = set_client_from_params(self.model_func(), parameters_to_weights(parameters))
        loss_tst, acc_tst = get_acc_loss(self.data_obj.tst_x, self.data_obj.tst_y,
                                         cur_cld_model, self.data_obj.dataset, 0)
        results['test_cld'] = {'loss': loss_tst, 'accuracy': acc_tst}
        loss_tst, acc_tst = get_acc_loss(self.cent_x, self.cent_y,
                                         cur_cld_model, self.data_obj.dataset, 0)
        results['cent_cld'] = {'loss': loss_tst, 'accuracy': acc_tst}

        if self.selected_clients_average_weight is not None:
            selected_client_model = set_client_from_params(self.model_func(), self.selected_clients_average_weight)
            loss_tst, acc_tst = get_acc_loss(self.data_obj.tst_x, self.data_obj.tst_y,
                                             selected_client_model, self.data_obj.dataset, 0)
            results['test_selected'] = {'loss': loss_tst, 'accuracy': acc_tst}
            loss_tst, acc_tst = get_acc_loss(self.cent_x, self.cent_y,
                                             selected_client_model, self.data_obj.dataset, 0)
            results['cent_selected'] = {'loss': loss_tst, 'accuracy': acc_tst}

        if self.all_client_average_weight is not None:
            all_client_model = set_client_from_params(self.model_func(), self.all_client_average_weight)
            loss_tst, acc_tst = get_acc_loss(self.data_obj.tst_x, self.data_obj.tst_y,
                                             all_client_model, self.data_obj.dataset, 0)
            results['test_all'] = {'loss': loss_tst, 'accuracy': acc_tst}
            loss_tst, acc_tst = get_acc_loss(self.cent_x, self.cent_y,
                                             all_client_model, self.data_obj.dataset, 0)
            results['cent_all'] = {'loss': loss_tst, 'accuracy': acc_tst}

        return results['test_cld']['loss'], results

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
