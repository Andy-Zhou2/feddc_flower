from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from typing import Callable, Union
import flwr as fl
import numpy as np
from collections import defaultdict
import numbers
from utils_general import set_client_from_params, get_acc_loss
from utils_dataset import DatasetObject
import torch

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
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        n_par: Optional[int] = None,
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
        self.state_gradient_diff = None
        self.n_par = n_par
        self.param_last_round = None
        self.config = dict() if config is None else config

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

        param_this_round = parameters_to_ndarrays(parameters)
        if server_round == 1:
            if self.n_par is None:
                self.n_par = param_this_round[0].size
            self.state_gradient_diff = [np.zeros(self.n_par)]
        else:
            self.state_gradient_diff = [param_this_round[0] - self.param_last_round[0]]
        self.param_last_round = param_this_round

        fit_configurations = []
        for client in clients:
            config = {
                'round_num': server_round,
                'model_func': self.model_func,
                'alpha': self.config['alpha'],
                'global_update_last': self.state_gradient_diff,
                'learning_rate': self.config['learning_rate'],
                'batch_size': self.config['batch_size'],
                'epoch': self.config['epoch'],
                'print_per': self.config['print_per'],
                'weight_decay': self.config['weight_decay'],
                'dataset_name': self.config['dataset_name'],
                'sch_step': 1,
                'sch_gamma': 1,
                'lr_decay_per_round': self.config['lr_decay_per_round'],
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

        print('aggregate fit')

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated = aggregate_weighted_average(fit_metrics)
        return parameters_aggregated, metrics_aggregated

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
        cur_cld_model = set_client_from_params(self.model_func(), parameters_to_ndarrays(parameters))
        loss_tst, acc_tst = get_acc_loss(self.data_obj.tst_x, self.data_obj.tst_y,
                                         cur_cld_model, self.data_obj.dataset, 0)
        return loss_tst, {'accuracy': acc_tst}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

