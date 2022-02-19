import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Preliminaries
import time
import glob
import math
import random
import copy
import shutil
import sys
#sys.path.append('./apex')
import typing as tp
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import collections.abc as container_abcs

from tqdm import tqdm
import yaml
from joblib import delayed, Parallel

# Visuals and CV2
import cv2

import pandas as pd
import numpy as np
from torch.cuda import amp
import matplotlib.pyplot as plt

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

#torch
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

#from cuml.neighbors import NearestNeighbors

from pytorch_metric_learning import losses

from dataset import SpectrogramDataset


ROOT = Path.cwd()
INPUT_ROOT = ROOT / "../../../../machine_sound" #"data" / "wav_data"


class CFG:
    seed = 42
    model_name = 'tf_efficientnet_b4_ns' #'efficientnet_b3' #efficientnet_b0-b7
    img_size = 224
    scheduler = 'CosineAnnealingLR'
    T_max = 10
    lr = 1e-5
    min_lr = 1e-6
    batch_size = 16
    weight_decay = 1e-6
    num_epochs = 50
    num_classes = 4
    embedding_size = 512
    n_fold = 0
    n_accumulate = 4
    temperature = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))

def get_loaders_for_training(
    args_dataset: tp.Dict, args_loader: tp.Dict,
    train_file_list: tp.List[str], val_file_list: tp.List[str]
):
    # # make dataset
    train_dataset = SpectrogramDataset(train_file_list, **args_dataset)
    val_dataset = SpectrogramDataset(val_file_list, **args_dataset)
    # # make dataloader
    train_loader = DataLoader(train_dataset, **args_loader["train"])
    val_loader = DataLoader(val_dataset, **args_loader["val"])

    return train_loader, val_loader, train_dataset, val_dataset

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, output_dir):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    history = defaultdict(list)
    scaler = amp.GradScaler()

    for step, epoch in enumerate(range(1,num_epochs+1)):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','valid']:
            if(phase == 'train'):
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluation mode

            running_loss = 0.0

            # Iterate over data
            for inputs,labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(CFG.device)
                labels = labels.to(CFG.device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast(enabled=True):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss = loss / CFG.n_accumulate

                    # backward only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()

                    # optimize only if in training phase
                    if phase == 'train' and (step + 1) % CFG.n_accumulate == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                        # zero the parameter gradients
                        optimizer.zero_grad()


                running_loss += loss.item()*inputs.size(0)

            epoch_loss = running_loss/dataset_sizes[phase]
            history[phase + ' loss'].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase=='valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = f"Model_{best_loss}_epoch_{epoch}.bin"
                torch.save(model.state_dict(), str(str(output_dir) + '/' + str( PATH)))
                
                plt.style.use('fivethirtyeight')
                plt.rcParams["font.size"] = "20"
                fig = plt.figure(figsize=(22,8))
                epochs = list(range(epoch))
                plt.plot(epochs, history['train loss'], label='train loss')
                plt.plot(epochs, history['valid loss'], label='valid loss')
                plt.ylabel('Loss', fontsize=20)
                plt.xlabel('Epoch', fontsize=20)
                plt.legend()
                plt.title('Loss Curve')
                plt.savefig(output_dir /'loss.png')

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss ",best_loss)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def run():

    # 1. Parse dataset and make train/ val folds

    # tmp_list = []
    # npz_files = glob.glob(str(INPUT_ROOT) + '/*/*')
    # for npz_file in npz_files:
    #     wav_fpath = npz_file
    #     wav_fname = npz_file.split('/')[-1]
    #     machine_db_type, operation_type, machine_id, file_id, channel_no = npz_file.split('/')[-1].split('-')
    #     machine_type = machine_db_type.split('_')[-1]
    #     tmp_list.append( [machine_type, machine_id, operation_type,
    #                                          wav_fname, wav_fpath])
    
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
    print('Test df shape ', val_df.shape)

    #TODO: Here we can select to use only normal or abnormal data for training

    train_file_list = train_df[["wav_file_path", "machine_type"]].values.tolist()
    val_file_list = val_df[["wav_file_path", "machine_type"]].values.tolist()

    print("train: {}, val: {}".format(len(train_file_list), len(val_file_list)))

    with open('../params/train_config_metricLearning.yaml') as settings_str:
        settings = yaml.safe_load(settings_str)

    for k, v in settings.items():
        print("[{}]".format(k))
        print(v)

    set_seed(settings["globals"]["seed"])
    device = torch.device(settings["globals"]["device"])
    output_dir = Path(settings["globals"]["output_dir"])

    # # # get loader
    train_loader, valid_loader, train_dataset, valid_dataset = get_loaders_for_training(
        settings["dataset"]["params"], settings["loader"], train_file_list, val_file_list)

    dataset_sizes = {
        'train' : train_df.shape[0],
        'valid' : val_df.shape[0]
    }
    dataloaders = {
        'train' : train_loader,
        'valid' : valid_loader
    }

    model = timm.create_model(CFG.model_name, pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, CFG.embedding_size)

    out = model(torch.randn(1, 3, CFG.img_size, CFG.img_size))
    print(f'Embedding shape: {out.shape}')

    model.to(CFG.device)

    criterion = SupervisedContrastiveLoss(temperature=CFG.temperature).to(CFG.device) # Custom Implementation
    # criterion = losses.SupConLoss(temperature=CFG.temperature).to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)

    num_epochs=CFG.num_epochs

    shutil.copy("../params/train_config_metricLearning.yaml", str(output_dir)+'/'+'train_config_metricLearning.yaml')
    shutil.copy("./train_metric_learning.py", str(output_dir)+'/'+'train_metric_learning.py')
    shutil.copy("./dataset_npy.py", str(output_dir)+'/'+'dataset.py')
    model, history = train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, output_dir)

    plt.style.use('fivethirtyeight')
    plt.rcParams["font.size"] = "20"
    fig = plt.figure(figsize=(22,8))
    epochs = list(range(CFG.num_epochs))
    plt.plot(epochs, history['train loss'], label='train loss')
    plt.plot(epochs, history['valid loss'], label='valid loss')
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(output_dir /'loss.png')

#     ############## Inference Using Trained model ###############
#     embeds = []
#     with torch.no_grad():
#         phase = "valid"
#         # Iterate over data
#         for inputs,labels in tqdm(dataloaders[phase]):
#             inputs = inputs.to(CFG.device)
#             labels = labels.to(CFG.device)

#             feat = model( inputs )
#             embeddings = feat.detach().cpu().numpy()
#             embeds.append(embeddings)

#     image_embeddings = np.concatenate(embeds)
#     print(f'Our image embeddings shape is {image_embeddings.shape}')
#     del embeds
#     gc.collect()

#     # Get neighbors from image_embeddings
#     # fit model
#     KNN = 50
#     knn_model = NearestNeighbors(n_neighbors = KNN)
#     knn_model.fit(image_embeddings)
#     distances, indices = knn_model.kneighbors(image_embeddings)

#     ###### Idea 1
#     # if we have both normal and anomaly samples in train foldself.
#     # and Knn neighbour of a test sample is abnormal (or normal)

#     ###### Idea 2
#     # If we have only normal in train fold
#     # then knn neighbour of test sample distance can be used to
#     # judge it as normal or abnormal ?

#     threshold = 20
#     for k in range(embeddings.shape[0]):
#         # Find the neighbour inside threshold n-dim ball
#         idx = np.where(distances[k,] < threshold)[0]
#         ids = indices[k,idx]
#         dist = distances[k, idx]
#         print(ids, dist)

if __name__ == "__main__":
    run()
