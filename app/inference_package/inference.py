import torch
import numpy as np

from inference_package.config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size,
                    hop_size, labels)
from inference_package.models import CNN, Transfer_Cnn14
from inference_package.inference_util import move_data_to_device
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

    def classify_audio_segment(self, audio_file, start_time, end_time):
        (waveform, _) = librosa.core.load(audio_file, sr=sample_rate,
                                          mono=True)
        waveform = waveform[int(start_time*sample_rate) : int(end_time*sample_rate), ]
        res = self.classify_waveform(waveform)
        return res

    def classify_audio_file(self, audio_file):
        """
        classify an audio file
        audio_file (str): audio path to classify
        """
        (waveform, _) = librosa.core.load(audio_file, sr=sample_rate,
                                          mono=True)

        res = self.classify_waveform(waveform)
        return res

    def classify_waveform(self, waveform):
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


def get_inference_audio(classifier, audio_file):
    probs = classifier.classify_audio_file(audio_file)

    index = np.argmax(probs)
    label = labels[index]
    prob = probs[index]

    second_largest_prob = 0

    for i in range(len(probs)):
        if probs[i] != prob and probs[i] > second_largest_prob:
            second_largest_prob= probs[i]
            second_larget_index = i
        else:
            continue
    second_label = labels[second_larget_index]
    return label, round(prob,3), second_label, round(second_largest_prob,3)

def get_inference_audio_segment(classifier, audio_file, start_time, end_time):
    probs = classifier.classify_audio_segment(audio_file, start_time, end_time)

    index = np.argmax(probs)
    label = labels[index]
    prob = probs[index]

    second_largest_prob = 0

    for i in range(len(probs)):
        if probs[i] != prob and probs[i] > second_largest_prob:
            second_largest_prob= probs[i]
            second_larget_index = i
        else:
            continue
    second_label = labels[second_larget_index]
    return label, round(prob,3), second_label, round(second_largest_prob,3)