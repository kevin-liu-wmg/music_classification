import os
import numpy as np
import argparse
import h5py
import librosa
import time


import config
from utilities import create_folder, traverse_folder, float32_to_int16


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0: max_len]


def pack_audio_files_to_hdf5(args):
    # Arguments & parameters
    workspace = args.workspace
    file_name = args.file_name
    input_txt = args.input_txt
    audio_path = args.audio_path

    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx

    # Paths
    with open(input_txt) as f:
        lines = f.readlines()
    song_list = [line.strip() for line in lines]
    audios_num = len(song_list)

    packed_hdf5_path = os.path.join(workspace, 'features', file_name)
    create_folder(os.path.dirname(packed_hdf5_path))

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name',
            shape=(audios_num,),
            dtype='S80')

        hf.create_dataset(
            name='waveform',
            shape=(audios_num, clip_samples),
            dtype=np.int16)

        hf.create_dataset(
            name='target',
            shape=(audios_num, classes_num),
            dtype=np.float32)

        for n in range(audios_num):
            line = song_list[n]
            genre_name = line.split('/')[0]
            genre_index = lb_to_idx[genre_name]

            audio_filename = os.path.join(audio_path, line)
            if os.path.isfile(audio_filename):
                (wav, fs) = librosa.core.load(audio_filename, sr=sample_rate,
                                                mono=True)
                wav = pad_truncate_sequence(wav, clip_samples)
                hf['audio_name'][n] = audio_filename.encode()
                hf['waveform'][n] = float32_to_int16(wav)
                hf['target'][n] = to_one_hot(genre_index, classes_num)

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument('--input_txt', type=str, required=True,
                                   help='Text file of input audios.')
    parser_pack_audio.add_argument('--audio_path', type=str, required=True,
                                   help='Path to audio files.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True,
                                   help='Directory of your workspace.')
    parser_pack_audio.add_argument('--file_name', type= str, required=True,
                                   help='The name of the hdf5 file')
    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)

    else:
        raise Exception('Incorrect arguments!')