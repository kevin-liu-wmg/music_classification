<div id="top"></div>

<!-- PROJECT LOGO -->
<div align="center">
  <h2 align="center">Genre classification using transfer learning on GTZAN dataset </h2>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#problem">Problem description</a></li>
    <li><a href="#data">Data description</a></li>
    <li><a href="#algorithm">Algorithm</a>
      <ul>
        <li><a href="#modeldescription">Model description</a></li>
        <li><a href="#modeltraining">Model training</a></li>
        <li><a href="#modelinference">Model inference</a></li>
        <li><a href="#modelperformance">Model performance</a></li>
      </ul>
    </li>
    <li><a href="#app">App description</a></li>
    <a href="#limitations">Limitations and future work</a></li>
  </ol>
</details>

<!-- Problem description -->
## Problem description
This repo reimplement the state-of-the-art work ([PANNs: Large-Scale Pretrained 
Audio Neural Networks for Audio Pattern Recognition] (https://arxiv.org/abs/1912.10211))
on music genre classification using the GTZAN dataset. 

<!-- Data description -->
## Data description
This dataset was used for the well known paper in genre classification 
'**Musical genre classification of audio signals**' by G. Tzanetakis and P. Cook
 in IEEE Transactions on Audio and Speech Processing 2002.
 
The dataset consists of 1,000 audio tracks each 30 seconds long. 
It contains 10 genres, each represented by 100 tracks. 
The tracks are all 22050Hz Mono 16-bit audio files in .wav format. 
More detailed information can be found [here](http://marsyas.info/downloads/datasets.html).

Codes below can be used to download the dataset as well as the splits on the dataset. 
```
!wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
!tar -zxvf genres.tar.gz
!wget https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/train_filtered.txt
!wget https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/valid_filtered.txt
!wget https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/test_filtered.txt
```

<!-- Algorithm -->
## Algorithm
Transfer learning is applied on the GTZAN dataset where the pretrained model was trained
on the audioset dataset. The model was proposed by [PANNs: Large-Scale Pretrained 
Audio Neural Networks for Audio Pattern Recognition] (https://arxiv.org/abs/1912.10211). 
### Model description
The model contains mel-spectrogram layer as well as CNN blockers. However, different 
model backbones can also be applied on this dataset as well, for example, the 
MobileNets, ResNets, VGGish, etc..  

Environment setup is required before running the model and app. 
```
conda create --name $YOUR_ENVIR_NAME python=3.7
conda activate $YOUR_ENVIR_NAME
pip install -r requirements.txt
```

### Model training
To train the model, we need to first store audio waveforms into hdf5 files. This 
improves the efficiency on training the model. We can use codes below to store waveforms 
and training our model. 
```
cd panns_transfer_to_gtzan/
audio_path=$YOUR AUDIO FILE ROOT
WORKSPACE=$YOUR WORKSPACE WHERE TO STORE CHECKPOINTS
train_txt=$PATH TO YOUR TRAIN TEXT FILE
train_file=$NAME OF YOUR HDF5 FILE TO SAVE WAVEFORM
valid_txt=$PATH TO YOUR VALID TEXT FILE
valid_file=$NAME OF YOUR HDF5 FILE TO SAVE WAVEFORM
python3 utils/features.py pack_audio_files_to_hdf5 --audio_path=$audio_path --workspace=$WORKSPACE
     --input_txt=$train_txt --file_name=$train_file
python3 utils/features.py pack_audio_files_to_hdf5 --audio_path=$audio_path --workspace=$WORKSPACE 
    --input_txt=$valid_txt --file_name=$valid_file
```

Then we can use the following codes for model training. 
```
PRETRAINED_CHECKPOINT_PATH=$YOUR PRETRAINED MODEL PATH
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train --workspace=$WORKSPACE 
    --model_type="Transfer_Cnn14" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH 
    --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 
    --epoch=50 --freeze_base --cuda
```
### Model inference
Model inference package is stored under panns_transfer_to_gtzan/inference. This part will 
be reimplemented in the future. 

### Model performance
The best accuracy on the valid dataset is about 0.76. However, better audio 
augmentations need to be implemented to improve the result and more advanced 
model architecture needs to be implemented as well.  

<!-- App -->
## App description
The app shows the dataset of GTZAN as well as the effect of applying the audio augmentations. 
In addition, inference can be made based on a valid YouTube link. 
```
streamlit run app.py
```

