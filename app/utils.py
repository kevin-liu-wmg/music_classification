import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import pandas as pd
import subprocess


def audio_augment(wave_file, noise, time, pitch):
    transform = []
    wav, sr = sf.read(wave_file)
    if noise:
        transform.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5))
    if time:
        transform.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5))
    if pitch:
        transform.append(PitchShift(min_semitones=-4, max_semitones=4, p=0.5))
    if len(transform) > 0:
        augment = Compose(transform)
        augmented_samples = augment(samples=wav, sample_rate=sr)
    else:
        augmented_samples = wav
    sf.write('audios/temp.wav', augmented_samples, sr, 'PCM_24')


def mix_up_audios(wave_file1, wave_file2, alpha):
    wav_1, sr1 = sf.read(wave_file1)
    wav_2, sr2 = sf.read(wave_file2)
    mix_up_wav = wav_1[0:sr1*30] * alpha + wav_2[0:sr2*30] * (1 - alpha)
    sf.write('audios/mixup.wav', mix_up_wav, sr1, 'PCM_24')


def create_numerical_table():
    model_back_bone = ['Basic CNN', 'CNN14', 'CNN14', 'CNN14', 'CNN14', 'CNN14']
    pretrained = ['False', 'False', 'True', 'True', 'True', 'True']
    freeze_base = ['Not applicable', 'Not applicable', 'False', 'True', 'False', 'True']
    augmentations = ['None', 'None', 'None', 'None', 'True', 'True']
    best_accuracy = [0.2843, 0.5279, 0.7595, 0.7157, 0.6345, 0.4315]
    df = pd.DataFrame(list(zip(model_back_bone, pretrained, freeze_base,
                               augmentations, best_accuracy)),
                      columns=['Model structure', 'Pretrained weights',
                               'Freeze Base', 'Audio augmentations', 'Best accuracy'])
    return df

def convert_video_to_audio():
    command = "ffmpeg -y -i video/video.mp4 -ab 160k -ac 2 -ar 22050 -vn audios/audio.wav"
    subprocess.call(command, shell=True)

