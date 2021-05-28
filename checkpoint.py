import os
import torch
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,
    "train_losses": [],
    "train_symbol_accuracy": [],
    "train_sentence_accuracy": [],
    "train_wer": [],
    "validation_losses": [],
    "validation_symbol_accuracy": [],
    "validation_sentence_accuracy": [],
    "validation_wer": [],
    "lr": [],
    "grad_norm": [],
    "model": {},
    "configs":{},
    "token_to_id":{},
    "id_to_token":{},
}


def save_checkpoint(checkpoint, dir="./checkpoints", prefix=""):
    # Padded to 4 digits because of lexical sorting of numbers.
    # e.g. 0009.pth
    filename = "{num:0>4}.pth".format(num=checkpoint["epoch"])
    if not os.path.exists(os.path.join(prefix, dir)):
        os.makedirs(os.path.join(prefix, dir))
    torch.save(checkpoint, os.path.join(prefix, dir, filename))


def load_checkpoint(path, cuda=use_cuda):
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)


def init_tensorboard(name="", base_dir="./tensorboard"):
    return SummaryWriter(os.path.join(name, base_dir))


def write_tensorboard(
    writer,
    epoch,
    grad_norm,
    train_loss,
    train_symbol_accuracy,
    train_sentence_accuracy,
    train_wer,
    validation_loss,
    validation_symbol_accuracy,
    validation_sentence_accuracy,
    validation_wer,
    model,
):
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_symbol_accuracy", train_symbol_accuracy, epoch)
    writer.add_scalar("train_sentence_accuracy",train_sentence_accuracy,epoch)
    writer.add_scalar("train_wer", train_wer, epoch)
    writer.add_scalar("validation_loss", validation_loss, epoch)
    writer.add_scalar("validation_symbol_accuracy", validation_symbol_accuracy, epoch)
    writer.add_scalar("validation_sentence_accuracy",validation_sentence_accuracy,epoch)
    writer.add_scalar("validation_wer",validation_wer,epoch)
    writer.add_scalar("grad_norm", grad_norm, epoch)

    for name, param in model.encoder.named_parameters():
        writer.add_histogram(
            "encoder/{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "encoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
            )

    for name, param in model.decoder.named_parameters():
        writer.add_histogram(
            "decoder/{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "decoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
            )
