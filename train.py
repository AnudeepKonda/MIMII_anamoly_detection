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
#import resnest.torch as resnest_torch

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
INPUT_ROOT = ROOT / "data/wav_data/"

'''
RAW_DATA = INPUT_ROOT / "birdsong-recognition"
TRAIN_AUDIO_DIR = RAW_DATA / "train_audio"
TRAIN_RESAMPLED_AUDIO_DIRS = [
  INPUT_ROOT / "birdsong-resampled-train-audio-{:0>2}".format(i)  for i in range(5)
]
TEST_AUDIO_DIR = RAW_DATA / "test_audio"
'''


tmp_list = []
for decibel_value in INPUT_ROOT.iterdir():
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

print(train_all.sample(5))
print('All df shape ', train_all.shape)

train_df, val_df = train_test_split(train_all, test_size=0.2)#, random_state=1213)

print('Train df shape ', train_df.shape)
print('Test df shape ', val_df.shape)


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


train_file_list = train_df[["wav_file_path", "machine_type"]].values.tolist()
val_file_list = val_df[["wav_file_path", "machine_type"]].values.tolist()

print("train: {}, val: {}".format(len(train_file_list), len(val_file_list)))

with open("./test_config.yaml") as settings_str:
    settings = yaml.safe_load(settings_str)

# for k, v in settings.items():
#     print("[{}]".format(k))
#     print(v)

# # # get loader
train_loader, val_loader, train_dataset, val_dataset = get_loaders_for_training(
    settings["dataset"]["params"], settings["loader"], train_file_list, val_file_list)

data, target = train_dataset.__getitem__(0)
print(data.shape)

# # for batch_idx, (data, target) in enumerate(train_loader):
# #     #data, target = data.to(device), target.to(device)
# #     print(data, target)
# train_iter = iter(train_loader)
# data, target = train_iter.next()
# print(data,target)
# print("END")
