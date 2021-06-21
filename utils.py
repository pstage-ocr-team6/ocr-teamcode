import yaml
from copy import deepcopy
import torch.optim as optim
from adamp import AdamP

from networks.Attention import Attention
from networks.SATRN import SATRN


def get_network(
    model_type,
    FLAGS,
    model_checkpoint,
    device,
    train_dataset,
):
    """Get network

    Args:
        model_type (str): Model name that wants to use.
        FLAGS (Flag): Configs of model.
        model_checkpoint (dict): model checkpoint.
        device (torch.device): Device type to use.
        train_dataset (list): train_dataset

    Returns:
            model : model
    """
    model = None

    if model_type == "SATRN":
        model = SATRN(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "CRNN":
        model = CRNN()
    elif model_type == "Attention":
        model = Attention(FLAGS, train_dataset, model_checkpoint).to(device)
    else:
        raise NotImplementedError

    return model


def get_optimizer(optimizer, params, lr, weight_decay=None):
    """Get Optimizer

    Args:
        optimizer (optimizer): optimizer.
        params (optimizer.params): optimizer.params
        lr (optimizer.lr): optimizer LR
        weight_decay (float, optional): weight decay (L2 penalty). Defaults to None.

    Returns:
        optimizer: optimizer
    """
    if optimizer == "AdamP":
        optimizer = AdamP(params, lr=lr)
    elif optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == "Adadelta":
        optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def get_wandb_config(config_file):
    """Get Wandb config from config_file

    Args:
        config_file (str): config_file path
    Returns:
        config (dict): original config
    """
    # load config file
    with open(config_file, 'r') as f:
        option = yaml.safe_load(f)
    config = deepcopy(option)

    # remove all except network
    keys = ["checkpoint", "input_size", "data", "optimizer", "wandb", "prefix"]
    for key in keys:
        del config[key]

    # modify some config key-value
    new_config = {
        "log_path": option['prefix'],
        "dataset_proportions": option['data']['dataset_proportions'],
        "test_proportions": option['data']['test_proportions'],
        "crop": option['data']['crop'],
        "rgb": "grayscale" if option['data']['rgb']==1 else "color",
        "input_size": (option['input_size']['height'], option['input_size']['width']),
        "optimizer": option['optimizer']['optimizer'],
        "learning_rate": option['optimizer']['lr'],
        "weight_decay": option['optimizer']['weight_decay'],
        "is_cycle": option['optimizer']['is_cycle'],
    }

    # merge
    config.update(new_config)

    # print log
    print("wandb save configs below:\n", list(config.keys()))

    return config 