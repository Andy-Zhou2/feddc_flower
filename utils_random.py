"""Randomness seeding script for FedDC.

Slightly adapted from the original https://github.com/camlsys/fl-project-template?tab=Apache-2.0-1-ov-file
Copyright 2024 Andy Zhou

This project is licensed under the MIT License.
"""
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from utils_general import device

import logging
import random
import numpy as np
import torch


"""A client manager that guarantees deterministic client sampling."""
class DeterministicClientManager(SimpleClientManager):
    """A deterministic client manager.

    Samples clients in the same order every time based on the seed. Also allows sampling
    with replacement.
    """

    def __init__(
            self,
            client_cid_generator: random.Random,
            enable_resampling: bool = False,
    ) -> None:
        """Initialize DeterministicClientManager.

        Parameters
        ----------
        client_cid_generator : random.Random
            A random number generator to generate client cids.
        enable_resampling : bool
            Whether to allow sampling with replacement.

        Returns
        -------
        None
        """
        super().__init__()

        self.client_cid_generator = client_cid_generator
        self.enable_resampling = enable_resampling

    def sample(
            self,
            num_clients: int,
            min_num_clients: int | None = None,
            criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        """Sample a number of Flower ClientProxy instances.

        Guarantees deterministic client sampling and enables
        sampling with replacement.

        Parameters
        ----------
        num_clients : int
            The number of clients to sample.
        min_num_clients : Optional[int]
            The minimum number of clients to sample.
        criterion : Optional[Criterion]
            A criterion to select clients.

        Returns
        -------
        List[ClientProxy]
            A list of sampled clients.
        """
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        cids = list(self.clients)

        if criterion is not None:
            cids = [cid for cid in cids if criterion.select(self.clients[cid])]
        # Shuffle the list of clients

        available_cids = []
        if num_clients <= len(cids):
            available_cids = self.client_cid_generator.sample(
                cids,
                num_clients,
            )
        elif self.enable_resampling:
            available_cids = self.client_cid_generator.choices(
                cids,
                k=num_clients,
            )
        else:
            log(
                logging.INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(cids),
                num_clients,
            )
            available_cids = []

        client_list = [self.clients[cid] for cid in available_cids]
        log(
            logging.INFO,
            "Sampled the following clients: %s",
            available_cids,
        )

        return client_list


def get_isolated_rng_tuple(seed: int, device: torch.device):
    """Get the random state for server/clients.

    Parameters
    ----------
    seed : int
        The seed.
    device : torch.device
        The device.

    Returns
    -------
    Tuple[random.Random, np.random.Generator,
          torch.Generator(CPU), Optional[torch.Generator(GPU)]]
        The random state for clients.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    torch_rng_cpu = torch.Generator()

    torch_rng_cpu.manual_seed(seed)
    torch_rng_gpu = torch.Generator(device=device) if device != "cpu" else None
    if torch_rng_gpu is not None:
        torch_rng_gpu.manual_seed(seed)

    return seed, rng, np_rng, torch_rng_cpu, torch_rng_gpu


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA-enabled GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_rng(seed: int):
    set_seed(seed)

    # Create server RNG tuple
    server_rng_tuple = get_isolated_rng_tuple(seed, device)
    server_random = server_rng_tuple[1]
    # Create client cid and seed generators
    client_cid_generator_rng = random.Random(server_random.randint(0, 2 ** 32 - 1))
    client_seed_generator_rng = random.Random(server_random.randint(0, 2 ** 32 - 1))
    log(logging.INFO, f"Using RNG seed: {seed}")
    return server_rng_tuple, client_cid_generator_rng, client_seed_generator_rng
