import torch.nn

from utils_libs import *
from utils_dataset import Dataset
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter

import time

max_norm = 10


# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0;
    loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval();
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


# --- Helper functions
def set_client_from_params(mdl, params):
    # if params of dim (1, D), then convert to (D,)
    if isinstance(params, list):
        assert len(params) == 1, "if params is NDArray, then it should be a list of length 1"
        params = params[0]
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(models: list[torch.nn.Module] | torch.nn.Module, n_par=None):
    if not isinstance(models, list):
        models = [models]
        flag_single = True
    else:
        flag_single = False
    if n_par == None:
        exp_mdl = models[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(models), n_par)).astype('float32')
    for i, mdl in enumerate(models):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    result = np.copy(param_mat)
    if flag_single:
        return result[0]
    else:
        return result


def train_model_FedDC(model, model_func, alpha, local_update_last, global_update_last, global_model_param, hist_i,
                      trn_x, trn_y,
                      learning_rate, batch_size, epoch, print_per,
                      weight_decay, dataset_name, sch_step, sch_gamma):
    # print(f'model: {model}')
    # print(f'model_func: {model_func}')
    # print(f'alpha: {alpha}')
    # print(f'local_update_last: {np.sum(local_update_last)}')
    # print(f'global_update_last: {np.sum(global_update_last)}')
    # print(f'global_model_param: {np.sum(global_model_param.numpy())}')
    # print(f'hist_i: {np.sum(hist_i)}')
    # print(f'trn_x: {np.sum(trn_x)}')
    # print(f'trn_y: {np.sum(trn_y)}')
    # print(f'learning_rate: {learning_rate}')
    # print(f'batch_size: {batch_size}')
    # print(f'epoch: {epoch}')
    # print(f'print_per: {print_per}')
    # print(f'weight_decay: {weight_decay}')
    # print(f'dataset_name: {dataset_name}')
    # print(f'sch_step: {sch_step}')
    # print(f'sch_gamma: {sch_gamma}')


    n_trn = trn_x.shape[0]
    state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device=device)
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]

            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                    # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

            loss_cp = alpha / 2 * torch.sum(
                (local_parameter - (global_model_param - hist_i)) * (local_parameter - (global_model_param - hist_i)))
            loss_cg = torch.sum(local_parameter * state_update_diff)

            loss = loss_f_i + loss_cp + loss_cg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay) / 2 * np.sum(params * params)
            print("loss_f_i: %.4f, loss_cp: %.4f, loss_cg: %.4f" % (loss_f_i, loss_cp, loss_cg.item()))
            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, scheduler.get_last_lr()[0]))
            print(
                f'local_update_last: {np.sum(local_update_last):.4f}, global_update_last: {np.sum(global_update_last):.4f}, '
                f'state_update_diff: {np.sum(state_update_diff.detach().numpy()):.4f}, local param: {np.sum(local_parameter.detach().numpy()):.4f}')

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def parameters_to_weights(parameters: Parameters) -> np.ndarray:
    return parameters_to_ndarrays(parameters)[0]

def weights_to_parameters(weights: np.ndarray) -> Parameters:
    return ndarrays_to_parameters([weights])