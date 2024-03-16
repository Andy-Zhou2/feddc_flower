import flwr as fl
from flwr.common import Metrics
from utils_dataset import DatasetObject
from utils_models import client_model
from utils_general import get_mdl_params, set_client_from_params

data_path = 'Folder/' # The folder to save Data & Model

# Generate IID or Dirichlet distribution
# IID
n_client = 100
data_obj = DatasetObject(dataset='mnist', n_client=n_client, seed=23, rule='iid', unbalanced_sgm=0, data_path=data_path)

class FeddcClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, n_par=None):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.n_par = n_par if n_par is not None else get_mdl_params([self.net])[0]

    def get_parameters(self, config):
        return get_mdl_params([self.net], self.n_par)[0]

    def fit(self, parameters, config):
        set_client_from_params(self.net, parameters)

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return loss, accuracy

    def get_metrics(self) -> Metrics:
        return {"accuracy": 0.0}

def client_fn(cid: str) -> fl.client.Client:
    """Load data and model for a specific client."""
    # Load model
    model_func = lambda : client_model(model_name)
    # Load data
    trainloader, testloader = load_data()
    # Create and return a client
    return FeddcClient(cid, model_func(), trainloader, testloader)