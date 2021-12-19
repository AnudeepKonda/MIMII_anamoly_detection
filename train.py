import os
import gc
import time
import shutil
import random
import warnings
import typing as tp
from pathlib import Path
from contextlib import contextmanager

from tqdm import tqdm
import yaml
from joblib import delayed, Parallel

import cv2
import librosa
import audioread
import soundfile as sf

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import resnest.torch as resnest_torch

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as ppe_extensions

from dataset import SpectrogramDataset

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
#   torch.backends.cudnn.deterministic = True  # type: ignore
#   torch.backends.cudnn.benchmark = True  # type: ignore


@contextmanager
def timer(name: str) -> None:
    """Timer Util"""
    t0 = time.time()
    print("[{}] start".format(name))
    yield
    print("[{}] done in {:.0f} s".format(name, time.time() - t0))

ROOT = Path.cwd()
INPUT_ROOT = ROOT / "../machine_sound"#"data" / "wav_data"

def get_loaders_for_training(
    args_dataset: tp.Dict, args_loader: tp.Dict,
    train_file_list: tp.List[str], val_file_list: tp.List[str]
):
    # # make dataset
    train_dataset = SpectrogramDataset(train_file_list, **args_dataset)
    val_dataset = SpectrogramDataset(val_file_list, **args_dataset)
    # # make dataloader
    train_loader = data.DataLoader(train_dataset, **args_loader["train"])
    val_loader = data.DataLoader(val_dataset, **args_loader["val"])

    return train_loader, val_loader, train_dataset, val_dataset

def get_model(args: tp.Dict):
    model =getattr(resnest_torch, args["name"])(pretrained=args["params"]["pretrained"])
    del model.fc
    #print(model)
    # # use the same head as the baseline notebook.
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, args["params"]["n_classes"]))

    return model

def train_loop(
    manager, args, model, device,
    train_loader, optimizer, scheduler, loss_func
):
    """Run minibatch training loop"""
    while not manager.stop_trigger:
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with manager.run_iteration():
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                #print(output.dtype, target.dtype)
                #output = output.to(device).float()
                target = target.type_as(output)
                #print(output.dtype, target.dtype)
                loss = loss_func(output, target)
                ppe.reporting.report({'train/loss': loss.item()})
                #print(loss)
                loss.backward()
                optimizer.step()
                scheduler.step()

def eval_for_batch(
    args, model, device,
    data, target, loss_func, eval_func_dict={}
):
    """
    Run evaliation for valid

    This function is applied to each batch of val loader.
    """
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    # Final result will be average of averages of the same size
    target = target.type_as(output)
    val_loss = loss_func(output, target).item()
    ppe.reporting.report({'val/loss': val_loss})

    for eval_name, eval_func in eval_func_dict.items():
        eval_value = eval_func(output, target).item()
        ppe.reporting.report({"val/{}".format(eval_aame): eval_value})

def set_extensions(
    manager, args, model, device, test_loader, optimizer,
    loss_func, eval_func_dict={}
):
    """set extensions for PPE"""

    my_extensions = [
        # # observe, report
        ppe_extensions.observe_lr(optimizer=optimizer),
        # ppe_extensions.ParameterStatistics(model, prefix='model'),
        # ppe_extensions.VariableStatisticsPlot(model),
        ppe_extensions.LogReport(),
        ppe_extensions.PlotReport(['train/loss', 'val/loss'], 'epoch', filename='loss.png'),
        ppe_extensions.PlotReport(['lr',], 'epoch', filename='lr.png'),
        ppe_extensions.PrintReport([
            'epoch', 'iteration', 'lr', 'train/loss', 'val/loss', "elapsed_time"]),
#         ppe_extensions.ProgressBar(update_interval=100),

        # # evaluation
        (
            ppe_extensions.Evaluator(
                test_loader, model,
                eval_func=lambda data, target:
                    eval_for_batch(args, model, device, data, target, loss_func, eval_func_dict),
                progress_bar=True),
            (1, "epoch"),
        ),
        # # save model snapshot.
        (
            ppe_extensions.snapshot(
                target=model, filename="snapshot_epoch_{.updater.epoch}.pth"),
            ppe.training.triggers.MinValueTrigger(key="val/loss", trigger=(1, 'epoch'))
        ),
    ]

    # # set extensions to manager
    for ext in my_extensions:
        if isinstance(ext, tuple):
            manager.extend(ext[0], trigger=ext[1])
        else:
            manager.extend(ext)

    return manager

if __name__ == "__main__":
    # 1. Parse dataset and make train/ val folds

    tmp_list = []
    for decibel_value in INPUT_ROOT.iterdir():
        if decibel_value.is_file():
            continue
        for machine in decibel_value.iterdir():
            if machine.is_file():
                continue
            machine_type = machine.stem
            print(f"Reading files in {machine_type} machine type")
            for id in machine.iterdir():
                if id.is_file():
                    continue
                id_type = id.stem
                print(f"Reading files in {id_type}")
                for operation in id.iterdir():
                    if operation.is_file():
                        continue
                    operation_type = operation.stem
                    assert operation_type in ["normal", "abnormal"], "Expected normal or abnormal"
                    for wav_f in operation.iterdir():
                        if wav_f.is_file() and wav_f.suffix == ".wav":
                            tmp_list.append( [machine_type, id_type, operation_type,
                                              wav_f.name, wav_f.as_posix()])

    train_all = pd.DataFrame(
        tmp_list, columns=["machine_type", "id_type", "operation_type",
                            "wav_filename", "wav_file_path"])

    print(train_all.sample(n=5, random_state=1))
    print('All df shape ', train_all.shape)

    train_df, test_df = train_test_split(train_all, test_size=0.15, random_state=1234)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=1234)
    
    print('Train df shape ', train_df.shape)
    print('Val df shape ', val_df.shape)
    
    # Filter normal samples
    train_df = train_df#train_df[train_df['operation_type'].isin(['normal'])]
    val_df = val_df#val_df[val_df['operation_type'].isin(['normal'])]
    
    print(train_df)
    print(val_df)

    print('Train df shape after filtering normal:', train_df.shape)
    print('Val df shape after filtering normal:', val_df.shape)

    #TODO: Here we can select to use only normal or abnormal data for training

    train_file_list = train_df[["wav_file_path", "machine_type"]].values.tolist()
    val_file_list = val_df[["wav_file_path", "machine_type"]].values.tolist()

    print("train: {}, val: {}".format(len(train_file_list), len(val_file_list)))

    with open('test_config.yaml') as settings_str:
        settings = yaml.safe_load(settings_str)

    for k, v in settings.items():
        print("[{}]".format(k))
        print(v)

    set_seed(settings["globals"]["seed"])
    device = torch.device(settings["globals"]["device"])
    output_dir = Path(settings["globals"]["output_dir"])
  
    # # # get loader
    train_loader, val_loader, train_dataset, val_dataset = get_loaders_for_training(
        settings["dataset"]["params"], settings["loader"], train_file_list, val_file_list)

    # # # get model
    model = get_model(settings["model"])
    model = model.to(device)

    # # # get optimizer
    optimizer = getattr(torch.optim,
                        settings["optimizer"]["name"])(model.parameters(), **settings["optimizer"]["params"])

    # # # get scheduler
    scheduler = getattr(
        torch.optim.lr_scheduler, settings["scheduler"]["name"]
    )(optimizer, **settings["scheduler"]["params"])

    # # # get loss
    loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])

    # # # create training manager
    trigger = None

    manager = ppe.training.ExtensionsManager(
        model, optimizer, settings["globals"]["num_epochs"],
        iters_per_epoch=int(train_dataset.__len__()/settings["loader"]["train"]["batch_size"]),
        stop_trigger=trigger,
        out_dir=output_dir
    )

    # # # set manager extensions
    manager = set_extensions(
        manager, settings, model, device,
        val_loader, optimizer, loss_func,
    )
    
    # # # copy relevant files into output_dir
    shutil.copy("./test_config.yaml", output_dir)
    shutil.copy("./train.py", output_dir)
    shutil.copy("./dataset.py", output_dir)


    '''
    for batch_idx, data in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        image, target = data
    '''
    # # runtraining
    train_loop(
        manager, settings, model, device,
        train_loader, optimizer, scheduler, loss_func)

