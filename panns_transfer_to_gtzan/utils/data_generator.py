import torch
import numpy as np
import h5py
from torch.utils import data
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)
from utilities import int16_to_float32
import config

class GTZANDataset(data.Dataset):
    def __init__(self, hdf5_path, is_augment, lb_to_ix):
        self.hdf5_path = hdf5_path
        self.lb_to_ix = lb_to_ix
        self.is_augment = is_augment
        if self.is_augment:
            self.num_samples = 22050 * 29
            self._get_augmentations()

        with h5py.File(self.hdf5_path, 'r') as hf:
            self.audio_names = hf['audio_name'][:]
            self.waveforms = hf['waveform'][:]
            self.targets = hf['target'][:]

    def _get_augmentations(self):
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([HighLowPass(sample_rate=22050)], p=0.8),
            RandomApply([Delay(sample_rate=22050)], p=0.5),
            RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=22050)], p=0.4),
            RandomApply([Reverb(sample_rate=22050)], p=0.3),
        ]
        self.augmentation = Compose(transforms=transforms)

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, index):
        audio_name = self.audio_names[index].decode()
        wav = int16_to_float32(self.waveforms[index])
        target = self.targets[index].astype(np.float32)
        if self.is_augment:
            wav = self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()

        data_dict = {
            'audio_name': audio_name,
            'waveform': wav,
            'target': target
        }
        return data_dict


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...},
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}

    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array(
            [data_dict[key] for data_dict in list_data_dict])

    return np_data_dict


# hdf5_path = "/home/ubuntu/workspaces/panns_transfer_to_gtzan/features/waveform.h5"
# data_train = GTZANDataset(hdf5_path=hdf5_path, is_augment=True, lb_to_ix=config.lb_to_idx)
# data_loader = data.DataLoader(dataset=data_train,
#                               batch_size=16,
#                               shuffle=True,
#                               drop_last=False,
#                               num_workers=4, collate_fn=collate_fn)
# iter_test_loader = iter(data_loader)
# data_test = next(iter_test_loader)
# print(data_test)