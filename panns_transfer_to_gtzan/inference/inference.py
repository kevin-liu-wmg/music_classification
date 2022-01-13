import os
import sys

import torch
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
sys.path.insert(1, os.path.join(sys.path[0], '../pytorch'))

from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size,
                    hop_size, labels)
from models import CNN, Transfer_Cnn14
from inference_util import move_data_to_device
import librosa


class ClassificationEngine(object):
    def __init__(self, model_type, model_path, device):

        if device >= 0:
            device = torch.device(f'cuda:{device}')
        else:
            device = torch.device(f'cpu')
        print(f'using device: {device}')

        if model_type == "CNN":
            model = CNN(num_channels=16, sample_rate=sample_rate, n_fft=1024,
                        f_min=fmin, f_max=fmax, num_mels=128, num_classes=10)
        else:
            Model = eval(model_type)
            model = Model(sample_rate, window_size, hop_size, mel_bins, fmin,
                          fmax, classes_num, False)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    def classify_audio_file(self, audio_file):
        """
        classify an audio file
        audio_file (str): audio path to classify
        """
        (waveform, _) = librosa.core.load(audio_file, sr=sample_rate,
                                          mono=True)
        if len(waveform) == 0:
            raise Exception("Audio file is empty.")
        else:
            waveform = waveform[None, :]
            waveform = move_data_to_device(waveform, self.device)

            with torch.no_grad():
                output_dict = self.model(waveform, None)

            clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()
            # embedding = output_dict['embedding'].data.cpu().numpy()
            probs = np.exp(clipwise_output[0])
            return probs


# model_type="Transfer_Cnn14"
# model_path="/home/ubuntu/workspaces/panns_transfer_to_gtzan/checkpoints/main/Transfer_Cnn14/pretrain=True/loss_type=clip_nll/augmentation=mixup/batch_size=32/freeze_base=False/BestAcc.pth"
# device=-1
# audio_file='/home/ubuntu/music_classification/data/genres/pop/pop.00002.wav'
# classifier=ClassificationEngine(model_type, model_path, device)
# probs=classifier.classify_audio_file(audio_file)
# index=np.argmax(probs)
# label = labels[index]
# print(label)
