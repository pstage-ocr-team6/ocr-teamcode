import os
import argparse
import multiprocessing
import numpy as np
import random
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from adamp import AdamP
import yaml
import wandb
from dotenv import load_dotenv
from tqdm import tqdm
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_tensorboard,
    write_tensorboard,
)
from psutil import virtual_memory

from flags import Flags
from utils import get_network, get_optimizer, get_wandb_config
from dataset import dataset_loader, START, PAD, load_vocab
from scheduler import CircularLRBeta

from metrics import word_error_rate, sentence_acc, get_worst_wer_img_path
from custom_augment import cutout, specAugment

# load env file
load_dotenv(verbose=True)


def id_to_string(tokens, data_loader, do_eval=0):
    result = []
    if do_eval:
        special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       data_loader.dataset.token_to_id["<EOS>"]]

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
                elif token == data_loader.dataset.token_to_id["<EOS>"]:
                    break
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result


def run_epoch(
    data_loader,
    model,
    epoch_text,
    criterion,
    optimizer,
    lr_scheduler,
    teacher_forcing_ratio,
    max_grad_norm,
    device,
    scaler=None,
    train=True,
    vis_wandb=True,
):
    # Disables autograd during validation mode
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()
        
    def _infer(input, expected):
        output = model(input, expected, train, teacher_forcing_ratio)
        decoded_values = output.transpose(1, 2)
        _, sequence = torch.topk(decoded_values, 1, dim=1)
        sequence = sequence.squeeze(1)

        loss = criterion(decoded_values, expected[:, 1:])
        return loss, sequence

    losses = []
    grad_norms = []
    correct_symbols = 0
    total_symbols = 0
    wer = 0
    num_wer = 0
    sent_acc = 0
    num_sent_acc = 0
    high_wer_imgs = []

    with tqdm(
        desc="{} ({})".format(epoch_text, "Train" if train else "Validation"),
        total=len(data_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for d in data_loader:
            torch.cuda.empty_cache()
            
            input = d["image"].to(device)

            # The last batch may not be a full batch
            curr_batch_size = len(input)
            expected = d["truth"]["encoded"].to(device)

            # Replace -1 with the PAD token
            expected[expected == -1] = data_loader.dataset.token_to_id[PAD]
            
            if scaler is not None:
                with autocast():
                    loss, sequence = _infer(input, expected)
            else:
                loss, sequence = _infer(input, expected)

            if train:
                optim_params = [
                    p
                    for param_group in optimizer.param_groups
                    for p in param_group["params"]
                ]
                optimizer.zero_grad()
                if scaler is not None:
                    # accumulates scaled gradients
                    scaler.scale(loss).backward()
                    
                    # clip gradients, it returns the total norm of all parameters
                    # unscale for gradient clipping
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        optim_params, max_norm=max_grad_norm
                    )
                    grad_norms.append(grad_norm)
                    
                    scaler.step(optimizer)
                    
                    # update scaler for next iteration
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        optim_params, max_norm=max_grad_norm
                    )
                    grad_norms.append(grad_norm)
                    optimizer.step()
                
                lr_scheduler.step()

            losses.append(loss.item())

            expected[expected == data_loader.dataset.token_to_id[PAD]] = -1
            expected_str = id_to_string(expected, data_loader, do_eval=1)
            sequence_str = id_to_string(sequence, data_loader, do_eval=1)
            wer += word_error_rate(sequence_str, expected_str)
            num_wer += 1
            sent_acc += sentence_acc(sequence_str, expected_str)
            num_sent_acc += 1
            correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            if not train and vis_wandb:
                max_wer_img_path, max_wer, gt_txt, pred_txt = get_worst_wer_img_path(d['path'], sequence_str, expected_str)
                high_wer_imgs.append(wandb.Image(
                    max_wer_img_path,
                    caption="Img: {}, WER: {:.4f} \n GT: {} \n Pred: {}".format(max_wer_img_path[-9:], max_wer, gt_txt, pred_txt)))

            pbar.update(curr_batch_size)

    if not train and vis_wandb:
        wandb.log({"high_wer_imgs": high_wer_imgs})

    expected = id_to_string(expected, data_loader)
    sequence = id_to_string(sequence, data_loader)
    print("-" * 10 + "GT ({})".format("train" if train else "valid"))
    print(*expected[:3], sep="\n")
    print("-" * 10 + "PR ({})".format("train" if train else "valid"))
    print(*sequence[:3], sep="\n")

    result = {
        "loss": np.mean(losses),
        "correct_symbols": correct_symbols,
        "total_symbols": total_symbols,
        "wer": wer,
        "num_wer": num_wer,
        "sent_acc": sent_acc,
        "num_sent_acc": num_sent_acc
    }
    if train:
        try:
            result["grad_norm"] = np.mean([tensor.cpu() for tensor in grad_norms])
        except:
            result["grad_norm"] = np.mean(grad_norms)

    return result


def main(config_file, on_cpu):
    """
    Train math formula recognition model
    """
    options = Flags(config_file).get()

    if options.wandb.wandb:
        # yaml to dict
        wandb_config = get_wandb_config(config_file)
        # write run name
        run_name = options.wandb.run_name if options.wandb.run_name else None
        # intialize wandb project
        wandb.init(project=os.getenv('PROJECT'), entity=os.getenv('ENTITY'), config=wandb_config, name=run_name)
        
    # make config
    with open(config_file, 'r') as f:
        option_dict = yaml.safe_load(f)

    # set random seed
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    is_cuda = torch.cuda.is_available() and not on_cpu
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    # Print system environments
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    if on_cpu:
        print(
            "[+] System environments\n",
            "The number of cpus : {}\n".format(num_cpus),
            "Memory Size : {}G\n".format(mem_size),
        )
    else:
        num_gpus = torch.cuda.device_count()
        print(
            "[+] System environments\n",
            "The number of gpus : {}\n".format(num_gpus),
            "The number of cpus : {}\n".format(num_cpus),
            "Memory Size : {}G\n".format(mem_size),
        )

    # Load checkpoint and print result
    checkpoint = (
        load_checkpoint(options.checkpoint, cuda=is_cuda)
        if options.checkpoint != ""
        else default_checkpoint
    )
    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        if checkpoint.get("train_score") is None:
            checkpoint["train_score"] = [
                0.9 * acc + 0.1 * wer 
                for acc, wer 
                in zip(checkpoint["train_sentence_accuracy"], checkpoint["train_wer"])
            ]
        if checkpoint.get("validation_score") is None:
            checkpoint["validation_score"] = [
                0.9 * acc + 0.1 * wer 
                for acc, wer 
                in zip(checkpoint["validation_sentence_accuracy"], checkpoint["validation_wer"])
            ]
            
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
            "Train Symbol Accuracy : {:.5f}\n".format(checkpoint["train_symbol_accuracy"][-1]),
            "Train Sentence Accuracy : {:.5f}\n".format(checkpoint["train_sentence_accuracy"][-1]),
            "Train WER : {:.5f}\n".format(checkpoint["train_wer"][-1]),
            "Train Score : {:.5f}\n".format(checkpoint["train_score"][-1]),
            "Train Loss : {:.5f}\n".format(checkpoint["train_losses"][-1]),
            "Validation Symbol Accuracy : {:.5f}\n".format(checkpoint["validation_symbol_accuracy"][-1]),
            "Validation Sentence Accuracy : {:.5f}\n".format(checkpoint["validation_sentence_accuracy"][-1]),
            "Validation WER : {:.5f}\n".format(checkpoint["validation_wer"][-1]),
            "Validation Score : {:.5f}\n".format(checkpoint["validation_score"][-1]),
            "Validation Loss : {:.5f}\n".format(checkpoint["validation_losses"][-1]),
        )

    # Get data
    train_transformed = transforms.Compose(
        [
            # Resize so all images have the same size
            # to_binary(),
            transforms.Resize((options.input_size.height, options.input_size.width)),
            transforms.RandomChoice([cutout(10,0.5,True,10),specAugment(row_num_masks=1,col_num_masks=1)]), # cutout, specAugment를 랜덤해서 고릅니다.
            transforms.ToTensor(),
        ]
    )
    val_transformed = transforms.Compose(
        [
            # Resize so all images have the same size
            # to_binary(),
            transforms.Resize((options.input_size.height, options.input_size.width)),
            transforms.ToTensor(),
        ]
    )
    train_data_loader, validation_data_loader, train_dataset, valid_dataset = dataset_loader(options, train_transformed, val_transformed)
    print(
        "[+] Data\n",
        "The number of train samples : {}\n".format(len(train_dataset)),
        "The number of validation samples : {}\n".format(len(valid_dataset)),
        "The number of classes : {}\n".format(len(train_dataset.token_to_id)),
    )

    # Get loss, model
    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        train_dataset,
    )
    criterion = model.criterion.to(device)

    # Get optimizer
    enc_params_to_optimise = [
        param for param in model.encoder.parameters() if param.requires_grad
    ]
    dec_params_to_optimise = [
        param for param in model.decoder.parameters() if param.requires_grad
    ]
    if options.optimizer.selective_weight_decay:
        no_decay_keywords = ['bias', 'norm.weight']
        params_to_optimise = [
            {
                'params': [param for name, param in model.named_parameters()
                        if param.requires_grad and not any(nd in name for nd in no_decay_keywords)],
                'weight_decay': options.optimizer.weight_decay,
            },
            {
                'params': [param for name, param in model.named_parameters()
                        if param.requires_grad and any(nd in name for nd in no_decay_keywords)],
                'weight_decay': 0.0,
            }
        ]
        if options.optimizer.optimizer == 'AdamP':
            optimizer = AdamP(
                params_to_optimise, lr=options.optimizer.lr,
            )
        else:
            optimizer = getattr(torch.optim, options.optimizer.optimizer)(
                params_to_optimise, lr=options.optimizer.lr,
            )
    else:
        params_to_optimise = [*enc_params_to_optimise, *dec_params_to_optimise]
        optimizer = get_optimizer(
            options.optimizer.optimizer,
            params_to_optimise,
            lr=options.optimizer.lr,
            weight_decay=options.optimizer.weight_decay,
        )
        
    print(
        "[+] Network\n",
        "Type: {}\n".format(options.network),
        "Encoder parameters: {}\n".format(
            sum(p.numel() for p in enc_params_to_optimise),
        ),
        "Decoder parameters: {} \n".format(
            sum(p.numel() for p in dec_params_to_optimise),
        ),
    )
        
    # load state dict to optimizer if needed
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = options.optimizer.lr
        
    # learning rate scheduler
    if options.optimizer.is_cycle:
        cycle = len(train_data_loader) * 100
        lr_scheduler = CircularLRBeta(
            optimizer, options.optimizer.lr, 10, 10, cycle, [0.95, 0.85]
        )
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=options.optimizer.lr_epochs,
            gamma=options.optimizer.lr_factor,
        )
        
    # Scaler for mixed precision
    scaler = GradScaler() if options.fp16 and not on_cpu else None

    # Log
    if not os.path.exists(options.prefix):
        os.makedirs(options.prefix)
    # log_file = open(os.path.join(options.prefix, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(options.prefix, "train_config.yaml"))
    if options.print_epochs is None:
        options.print_epochs = options.num_epochs
    writer = init_tensorboard(name=options.prefix.strip("-"))
    start_epoch = checkpoint["epoch"]
    train_symbol_accuracy = checkpoint["train_symbol_accuracy"]
    train_sentence_accuracy = checkpoint["train_sentence_accuracy"]
    train_wer = checkpoint["train_wer"]
    train_score = checkpoint["train_score"]
    train_losses = checkpoint["train_losses"]
    validation_symbol_accuracy = checkpoint["validation_symbol_accuracy"]
    validation_sentence_accuracy = checkpoint["validation_sentence_accuracy"]
    validation_wer = checkpoint["validation_wer"]
    validation_score = checkpoint["validation_score"]
    validation_losses = checkpoint["validation_losses"]
    learning_rates = checkpoint["lr"]
    grad_norms = checkpoint["grad_norm"]

    # Load or initialize best score
    if options.checkpoint:
        if checkpoint.get("best_score") is not None:
            best_score = checkpoint["best_score"]
        else:
            best_validation_score = max(validation_score)
            best_epoch = validation_score.index(best_validation_score)
            best_score = {
                'epoch': best_epoch + 1, 
                'score': best_validation_score, 
                'sentence_accuracy': validation_sentence_accuracy[best_epoch], 
                'wer': validation_wer[best_epoch], 
                'symbol_accuracy': validation_symbol_accuracy[best_epoch],
            }
    else:
        best_score = {
            'epoch': 0, 
            'score': 0, 
            'sentence_accuracy': 0, 
            'wer': 1e9, 
            'symbol_accuracy': 0,
        }
    no_increase = 0
    
    # initial teacher_forcing_ratio
    teacher_forcing_ratio = options.teacher_forcing_ratio
    
    # Train
    for epoch in range(options.num_epochs):
        if options.patience >= 0 and no_increase > options.patience:
            break
        
        start_time = time.time()

        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=options.num_epochs,
            epoch=start_epoch + epoch + 1,
            pad=len(str(options.num_epochs)),
        )
        
        # linear teacher forcing scheduling
        if options.teacher_forcing_damp > 0 and teacher_forcing_ratio > 0:
            teacher_forcing_ratio = max(teacher_forcing_ratio - options.teacher_forcing_damp, 0)

        # Train
        train_result = run_epoch(
            train_data_loader,
            model,
            epoch_text,
            criterion,
            optimizer,
            lr_scheduler,
            options.teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            scaler=scaler,
            train=True,
        )

        train_losses.append(train_result["loss"])
        grad_norms.append(train_result["grad_norm"])
        train_epoch_symbol_accuracy = (train_result["correct_symbols"] / train_result["total_symbols"])
        train_symbol_accuracy.append(train_epoch_symbol_accuracy)
        train_epoch_sentence_accuracy = (train_result["sent_acc"] / train_result["num_sent_acc"])

        train_sentence_accuracy.append(train_epoch_sentence_accuracy)
        train_epoch_wer = (train_result["wer"] / train_result["num_wer"])
        train_wer.append(train_epoch_wer)
        train_epoch_score = ((float(train_epoch_sentence_accuracy) * 0.9) + ((1 - float(train_epoch_wer)) * 0.1))
        train_score.append(train_epoch_score)
        epoch_lr = lr_scheduler.get_lr()  # cycle

        # Validation
        validation_result = run_epoch(
            validation_data_loader,
            model,
            epoch_text,
            criterion,
            optimizer,
            lr_scheduler,
            options.teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            scaler=scaler,
            train=False,
            vis_wandb=True,
        )
        validation_losses.append(validation_result["loss"])
        validation_epoch_symbol_accuracy = (validation_result["correct_symbols"] / validation_result["total_symbols"])
        validation_symbol_accuracy.append(validation_epoch_symbol_accuracy)

        validation_epoch_sentence_accuracy = (
                validation_result["sent_acc"] / validation_result["num_sent_acc"]
        )
        validation_sentence_accuracy.append(validation_epoch_sentence_accuracy)
        validation_epoch_wer = (
                validation_result["wer"] / validation_result["num_wer"]
        )
        validation_wer.append(validation_epoch_wer)
        validation_epoch_score = (
                (float(validation_epoch_sentence_accuracy) * 0.9) + ((1 - float(validation_epoch_wer)) * 0.1)
        )
        validation_score.append(validation_epoch_score)
        
        # things to save
        elapsed_time = time.time() - start_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        checkpoint_log = {
            "epoch": start_epoch + epoch + 1,
            "train_losses": train_losses,
            "train_symbol_accuracy": train_symbol_accuracy,
            "train_sentence_accuracy": train_sentence_accuracy,
            "train_wer": train_wer,
            "train_score": train_score,
            "validation_losses": validation_losses,
            "validation_symbol_accuracy": validation_symbol_accuracy,
            "validation_sentence_accuracy": validation_sentence_accuracy,
            "validation_wer": validation_wer,
            "validation_score": validation_score,
            "lr": epoch_lr,
            "elapsed_time": elapsed_time,
            "grad_norm": grad_norms,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "configs": option_dict,
            "token_to_id": train_data_loader.dataset.token_to_id,
            "id_to_token": train_data_loader.dataset.id_to_token,
            "best_score": best_score,
        }
                
        # update best score & save checkpoint
        if validation_epoch_score > best_score['score']:
            best_score = {
                'epoch': start_epoch + epoch + 1, 
                'score': validation_epoch_score, 
                'sent_acc': validation_epoch_sentence_accuracy, 
                'wer': validation_epoch_wer, 
                'sym_acc': validation_epoch_symbol_accuracy,
            }
            save_checkpoint(checkpoint_log, prefix=options.prefix)
            no_inrease = 0
        else:
            if not options.save_best_only:
                save_checkpoint(checkpoint_log, prefix=options.prefix)
            no_inrease += 1
            
        if options.wandb.wandb:
            wandb_log = {}
            # remove useless log
            removed_keys = ['model', 'optimizer', 'configs', 'epoch', 'token_to_id', 'id_to_token', 'best_score']
            for key in removed_keys:
                del checkpoint_log[key]
            # convert key-value datatype to int
            for key, value in checkpoint_log.items():
                if isinstance(value, list):
                    wandb_log[key] = value[-1]
                else:
                    wandb_log[key] = value
            # save log in wandb
            wandb.log(wandb_log)

        # Summary
        if epoch % options.print_epochs == 0 or epoch == options.num_epochs - 1:
            output_string = (
                "{epoch_text}: "
                "Train Symbol Accuracy = {train_symbol_accuracy:.5f}, "
                "Train Sentence Accuracy = {train_sentence_accuracy:.5f}, "
                "Train WER = {train_wer:.5f}, "
                "Train Score = {train_score:.5f}, "
                "Train Loss = {train_loss:.5f}, "
                "Validation Symbol Accuracy = {validation_symbol_accuracy:.5f}, "
                "Validation Sentence Accuracy = {validation_sentence_accuracy:.5f}, "
                "Validation WER = {validation_wer:.5f}, "
                "Validation Score = {validation_score:.5f}, "
                "Validation Loss = {validation_loss:.5f}, "
                "Best Model Epoch = {best_epoch}, "
                "lr = {lr} "
                "(time elapsed {time})"
            ).format(
                epoch_text=epoch_text,
                train_symbol_accuracy=train_epoch_symbol_accuracy,
                train_sentence_accuracy=train_epoch_sentence_accuracy,
                train_wer=train_epoch_wer,
                train_score=train_epoch_score,
                train_loss=train_result["loss"],
                validation_symbol_accuracy=validation_epoch_symbol_accuracy,
                validation_sentence_accuracy=validation_epoch_sentence_accuracy,
                validation_wer=validation_epoch_wer,
                validation_score=validation_epoch_score,
                validation_loss=validation_result["loss"],
                best_epoch=best_score["epoch"],
                lr=epoch_lr,
                time=elapsed_time_str,
            )
            print(output_string)
            with open(os.path.join(options.prefix, "log.txt"), 'a') as f:
                f.write(output_string+ "\n")
            write_tensorboard(
                writer,
                start_epoch + epoch + 1,
                train_result["grad_norm"],
                train_result["loss"],
                train_epoch_symbol_accuracy,
                train_epoch_sentence_accuracy,
                train_epoch_wer,
                train_epoch_score,
                validation_result["loss"],
                validation_epoch_symbol_accuracy,
                validation_epoch_sentence_accuracy,
                validation_epoch_wer,
                validation_epoch_score,
                model,
            )
        
    
    best_score = {' '.join(k.split('_')).title(): v for k, v in best_score.items()}
    print(f"\nBEST MODEL:\n{best_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        default="configs/SATRN.yaml",
        type=str,
        help="Path of configuration file",
    )
    parser.add_argument("--cpu", action="store_true")
    
    parser = parser.parse_args()
    main(parser.config_file, parser.cpu)
