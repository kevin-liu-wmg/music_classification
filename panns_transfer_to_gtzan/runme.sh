#!bin/bash

DATASET_DIR="/home/ubuntu/music_classification/data/genres"
WORKSPACE="/home/ubuntu/workspaces/panns_transfer_to_gtzan"

python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

PRETRAINED_CHECKPOINT_PATH="/home/ubuntu/music_classification/panns_transfer_to_gtzan/pretrained_model/Cnn14_mAP=0.431.pth"

CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Cnn14" --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda

CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda

DATASET_DIR="/home/ubuntu/music_classification/data/genres"
WORKSPACE="/home/ubuntu/workspaces/panns_transfer_to_gtzan"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main_new.py train --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=16 --epoch=5 --freeze_base --cuda

#####
MODEL_TYPE="Transfer_Cnn13"
PRETRAINED_CHECKPOINT_PATH="/vol/vssp/msos/qk/bytedance/workspaces_important/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/660000_iterations.pth"
python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --freeze_base --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --few_shots=10 --random_seed=1000 --resume_iteration=0 --stop_iteration=10000 --cuda

python3 utils/plot_statistics.py 1 --workspace=$WORKSPACE --select=2_cnn13


########### NEW

audio_path="/home/ubuntu/music_classification/data/genres"
WORKSPACE="/home/ubuntu/workspaces/panns_transfer_to_gtzan"
input_txt="/home/ubuntu/music_classification/data/train_filtered.txt"
file_name="train.h5"
python3 utils/features.py pack_audio_files_to_hdf5 --audio_path=$audio_path --workspace=$WORKSPACE --input_txt=$input_txt --file_name=$file_name


audio_path="/home/ubuntu/music_classification/data/genres"
WORKSPACE="/home/ubuntu/workspaces/panns_transfer_to_gtzan"
input_txt="/home/ubuntu/music_classification/data/valid_filtered.txt"
file_name="valid.h5"
python3 utils/features.py pack_audio_files_to_hdf5 --audio_path=$audio_path --workspace=$WORKSPACE --input_txt=$input_txt --file_name=$file_name

WORKSPACE="/home/ubuntu/workspaces/panns_transfer_to_gtzan"
PRETRAINED_CHECKPOINT_PATH="/home/ubuntu/music_classification/panns_transfer_to_gtzan/pretrained_model/Cnn14_mAP=0.431.pth"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --workspace=$WORKSPACE --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --epoch=50 --freeze_base --cuda

# pretrained freeze_base CNN14 "mix-up" "audio-augment"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --workspace=$WORKSPACE --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --epoch=50 --cuda --audio_augment --freeze_base

# pretrained CNN14 "mix-up" "audio-augment"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --workspace=$WORKSPACE --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --epoch=50 --cuda --audio_augment


# pretrained freeze_base CNN14
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --workspace=$WORKSPACE --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='none' --learning_rate=1e-4 --batch_size=32 --epoch=50 --cuda --freeze_base

# pretrained  CNN14
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --workspace=$WORKSPACE --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='none' --learning_rate=1e-4 --batch_size=32 --epoch=50 --cuda

# CNN14
PRETRAINED_CHECKPOINT_PATH=""
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --workspace=$WORKSPACE --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='none' --learning_rate=1e-4 --batch_size=32 --epoch=50 --cuda

# CNN
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --workspace=$WORKSPACE --model_type="CNN" --loss_type=clip_nll --augmentation='none' --learning_rate=1e-4 --batch_size=32 --epoch=50 --cuda
