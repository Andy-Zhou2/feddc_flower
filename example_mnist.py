from utils_dataset import DatasetObject
from utils_models import client_model
from utils_general import get_mdl_params, parameters_to_weights, weights_to_parameters
from client import client_fn

import torch
import flwr as fl
from strategy_feddc import FedDC

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


@hydra.main(config_path="Configs", config_name="config")
def main(cfg: DictConfig):
    data_path = to_absolute_path(cfg.data.data_path)
    print(data_path)
    n_client = cfg.data.n_client

    data_obj = DatasetObject(dataset=cfg.data.dataset, n_client=n_client, seed=cfg.data.seed,
                             rule=cfg.dataset_partition.rule,
                             rule_arg=cfg.dataset_partition.rule_arg,
                             unbalanced_sgm=cfg.dataset_partition.unbalanced_sgm,
                             data_path=data_path)


    model_name = cfg.model.model_name

    # Common hyperparameters
    com_amount = cfg.fl.num_total_rounds
    weight_decay = cfg.fl.weight_decay
    batch_size = cfg.fl.batch_size
    act_prob = cfg.fl.participation_ratio
    lr_decay_per_round = cfg.fl.lr_decay_per_round

    # Model function
    def model_func():
        return client_model(model_name)

    torch.manual_seed(37)

    init_model = model_func()
    init_weights = get_mdl_params(init_model)
    n_par = init_weights.size

    epoch = cfg.fl.epoch_per_round_per_client
    alpha_coef = cfg.fl.alpha_coef
    learning_rate = cfg.fl.learning_rate
    print_per = cfg.fl.print_per

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
        client_resources={"num_cpus": cfg.resources.num_cpu, "num_gpus": cfg.resources.num_gpu},
    )


if __name__ == "__main__":
    main()
