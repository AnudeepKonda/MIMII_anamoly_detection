import typing as tp
from pathlib import Path
import cv2
import librosa
import numpy as np
import pandas as pd
import random
import soundfile as sf
import torch.utils.data as data

import matplotlib.pyplot as plt
import random
random.seed(111)

MACHINE_CODE = {
    'pump': 0, 'valve': 1, 'slider': 2, 'fan': 3
}

INV_MACHINE_CODE = {v: k for k, v in MACHINE_CODE.items()}

PERIOD = 10

# implementation of SpecAugment paper here, without time warping
# set percentage of frames to mask so should work with long and short segments.
def spec_augment(spec: np.ndarray, num_mask=2,
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0

    return spec

class SpectrogramDataset(data.Dataset):
    def __init__(self,
                 file_list: tp.List[tp.List[str]],
                 is_val=False,
                 img_size=224,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 microphone_id=0,
                 melspectrogram_parameters={}, metric_learning=False):

        self.file_list = file_list  # list of list: [file_path, emachine_code]
        self.img_size = img_size
        self.microphone_id = microphone_id
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters

        self.n_mels = 128
        self.frames = 5
        self.n_fft = 2048
        self.hop_length = 512
        self.is_val = is_val

        self.metric_learning = metric_learning

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        wav_path, emachine_code = self.file_list[idx]
        #sample = self.df.loc[idx, :]
        #wav_name = sample["wav_filename"]
        #machine_code = sample["machine_type"]

        y, sr = np.load( wav_path )['arr_0'], 16000  
        images = []
        
        # resample 22050 to 16000
        y = librosa.istft(y.astype(np.float32), hop_length=512, win_length=2048, window='hann', center=True, dtype=None, length=220500)
        y = librosa.resample(y, 22050, 16000, res_type='soxr_vhq')
        spec = librosa.stft(y, n_fft=2048, hop_length=512)
        spec = spec.astype(np.float16)

        power_spec = np.abs(spec)**2
        melspec = librosa.feature.melspectrogram(S=power_spec, sr=16000, n_mels=128)

        if not self.is_val:
            if self.spectrogram_transforms:
                #melspec = self.spectrogram_transforms(melspec)
                prob = random.uniform(0, 1)
                if prob <= 0.5:   
                    melspec = spec_augment(melspec)
            else:
                pass

        # Is this necessary
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)
        images.append(image)

        # 1-hot encoding of labels
        labels = np.zeros(len(MACHINE_CODE), dtype=int)
        labels[MACHINE_CODE[emachine_code]] = 1
        
        if self.is_val:
            # get binary label
            if 'abnormal' in wav_path:
                binary_label = np.array([0., 1.])
            else:
                binary_label = np.array([1., 0.])
            
            return np.array(images[0]), [MACHINE_CODE[emachine_code], binary_label]

        return np.array(images[0]), MACHINE_CODE[emachine_code]

def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V