import streamlit as st
import os

import glob
from PIL import Image
from utils import audio_augment, mix_up_audios, create_numerical_table, convert_video_to_audio
from inference_package.inference import get_inference_audio, get_inference_audio_segment
from pytube import YouTube
from streamlit_player import st_player
from config import audio_root_path, classifier, genres

st.set_page_config(page_title="Genre Classifier", page_icon="🎶", layout="wide")

st.header("Genre classification using transfer learning on GTZAN dataset")
st.markdown("> 01/12/2022")



with st.expander("GTZAN dataset"):
    st.markdown(
        "This dataset was used for the well known paper in genre classification '**Musical genre classification of audio signals**' "
        "by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.")
    st.markdown(
        "The dataset consists of 1,000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. "
        "The tracks are all 22050Hz Mono 16-bit audio files in .wav format. "
        "More detailed information can be found [here](http://marsyas.info/downloads/datasets.html).",
        unsafe_allow_html=True)
    with st.markdown("GTZAN music player"):
        # st.file_uploader('please choose a file')
        placeholder = st.empty()
        selected_file = None
        with placeholder.container():
            st1, st2 = st.columns(2)
            with st1:
                selected_genre = st.selectbox('Genre of choice', [""] + genres)
                if selected_genre:
                    list_files = glob.glob(os.path.join(audio_root_path, selected_genre, '*.wav'))
            with st2:
                if selected_genre:
                    list_files = [file.split('/')[-1].split('.')[1] + '.wav' for file in list_files]
                    selected_file = st.selectbox('select a file', list_files)
                    selected_file = os.path.join(audio_root_path, selected_genre, selected_genre + '.' + selected_file)
            if selected_file:
                st.audio(selected_file)
st.markdown("---", unsafe_allow_html=True)


with st.expander("Transfer learning and audio augmentation"):
    st.markdown(
        "> _Transfer learning:_")
    col1, mid, col2 = st.columns([10,1,10])
    transfer_learning_image = Image.open('image/transfer_learning.jpeg')
    with col1:
        st.image(transfer_learning_image, caption='Transfer learning')
    with col2:
        st.write('> _What is Transfer learning_: '
                 'Transfer learning is a machine learning technique where '
                 'a model trained on one task is '
                 're-purposed on a second related task.')
        st.write('> _Benefits of transfer learning_:')
        st.write("""
                    - Removing the need for a large set of labelled training 
                    data for every new model.
                    - Improving the efficiency of machine learning development 
                    and deployment for multiple models.
                """)
        st.write('> _Transfer learning on GTZAN:_')
        st.write('Advanced models are trained on the AudioSet dataset (over 2 '
                 'millions audios of around 1000 classes). Then the model '
                 'weights are preserved to train the GTZAN dataset'
                 ' (1000 audios with 10 classes). This work is published at '
                 'PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition'
                 'by Qiuqiang Kong. etc. 2020.')
    st.markdown(
        "> _Audio augmentation_",unsafe_allow_html=True)
# st.markdown("---", unsafe_allow_html=True)
    with st.markdown('Raw audio augmentation'):
        placeholder = st.empty()
        with placeholder.container():
            st3, st4 = st.columns([10, 10])
            with st3:
                selected_genre_2 = st.selectbox('Genre of Choice', [""] + genres)
                if selected_genre_2:
                    list_files = glob.glob(
                         os.path.join(audio_root_path, selected_genre_2, '*.wav'))
                    list_files = [file.split('/')[-1].split('.')[1] + '.wav' for file in
                               list_files]
                    selected_file_2 = st.selectbox('Select a file', list_files)
                    selected_file = os.path.join(audio_root_path, selected_genre_2,
                                             selected_genre_2 + '.' + selected_file_2)
                if selected_file:
                    st.audio(selected_file)
            with st4:
                if selected_file:
                    noise = st.checkbox('Insert Gaussian noises')
                    time = st.checkbox("Time Stretch")
                    pitch = st.checkbox("Pitch swift")
                    audio_augment(selected_file, noise, time, pitch)
                    if noise or time or pitch:
                        st.audio('audios/temp.wav')
        # st.markdown("---", unsafe_allow_html=True)
    st.markdown('> _Mix-up augmentations_')
    st.latex(r'''
        X^{\prime} = \alpha * X_{1} + (1 - \alpha) * X_{2} \\
        Y^{\prime} = \alpha * Y_{1} + (1 - \alpha) * Y_{2} 
    ''')
    with st.markdown('Audio mix-up augmentation'):
        placeholder = st.empty()
        with placeholder.container():
            st3, st4 = st.columns([10, 10])
            song_1 = None
            song_2 = None
            with st3:
                song_1_genre = st.selectbox('Genre of first song',
                                                [""] + genres)
                if song_1_genre:
                    list_files = glob.glob(
                        os.path.join(audio_root_path, song_1_genre,
                                     '*.wav'))
                    list_files = [file.split('/')[-1].split('.')[1] + '.wav' for
                                  file in
                                  list_files]
                    song_1_file = st.selectbox('Select the first song', list_files)
                    song_1 = os.path.join(audio_root_path,
                                                 song_1_genre,
                                                 song_1_genre + '.' + song_1_file)
                if song_1:
                    st.audio(song_1)
            with st4:
                song_2_genre = st.selectbox('Genre of second song',
                                                [""] + genres)
                if song_2_genre:
                    list_files = glob.glob(
                        os.path.join(audio_root_path, song_2_genre,
                                     '*.wav'))
                    list_files = [file.split('/')[-1].split('.')[1] + '.wav' for
                                  file in
                                  list_files]
                    song_2_file = st.selectbox('Select the second song', list_files)
                    song_2 = os.path.join(audio_root_path,
                                                 song_2_genre,
                                                 song_2_genre + '.' + song_2_file)
                if song_2:
                    st.audio(song_2)

    with st.markdown('Audio mix-up augmentation'):
        placeholder = st.empty()
        alpha = 0
        with placeholder.container():
            left, right = st.columns([10, 10])
            with left:
                if song_1 and song_2:
                    alpha = st.number_input('Please enter a number between 0 and 1')
            with right:
                if alpha > 0 and alpha < 1:
                    mix_up_audios(song_1, song_2, alpha)
                    st.audio('audios/mixup.wav')


# st.markdown("---", unsafe_allow_html=True)

with st.expander("Model performance"):
    st.markdown(
        "> _DISCLAIMER_: Models are trained on the same epoch numbers (50) "
        "with the same learning rate and loss function")
    df = create_numerical_table()
    st.table(df)
    st.markdown(
        "**Tensorboard** can be accessed [here](http://localhost:6006/ )",
        unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

with st.expander("Real time inference"):
    with st.markdown('Inference audio genre'):
        placeholder = st.empty()
        with placeholder.container():
            st5, st6 = st.columns([10, 10])
            audio = None
            with st5:
                audio_genre = st.selectbox('Genre of selected audio',
                                                [""] + genres)
                if audio_genre:
                    list_files = glob.glob(
                        os.path.join(audio_root_path, audio_genre,
                                     '*.wav'))
                    list_files = [file.split('/')[-1].split('.')[1] + '.wav' for
                                  file in
                                  list_files]
                    audio_file = st.selectbox('Select a song', list_files)
                    audio = os.path.join(audio_root_path,
                                                 audio_genre,
                                                 audio_genre + '.' + audio_file)
                if audio:
                    st.audio(audio)
            with st6:
                submit = st.button('Inference')
                if submit:
                    with st.spinner(text="This may take a moment..."):
                        largest_label, largest_prob, second_label, second_prob = get_inference_audio(classifier, audio)
                    st.write(f'Predicted label is {largest_label} '
                             f'with probability {largest_prob}')
                    st.write(f'The second possible genre could be {second_label} '
                             f'with probability {second_prob}')

st.markdown("---", unsafe_allow_html=True)
with st.expander("Fun part"):
    st.write('Download your favorite music from YouTube')
    url = st.text_input("Enter the URL of the video to download")
    if url:
        with st.spinner("Downloading..."):
            st_player(url)
            button = st.button("Download")
            if button:
                file_ = YouTube(url).streams.filter(progressive=True,
                                                    file_extension="mp4").first().download(filename='video/video.mp4')
                st.success("Downloaded")

        with st.spinner(text="This may take a moment..."):
            button_2 = st.button("Get Result")
            if button_2:
                convert_video_to_audio()
                largest_label, largest_prob, second_label, second_prob = get_inference_audio(classifier,
                    'audios/audio.wav')
                st.success("Finished")
                st.markdown('The entire music is predicted with labels as follows:')
                st.write(f'Predicted label is {largest_label} '
                         f'with probability {largest_prob}')
                st.write(f'The second possible genre could be {second_label} '
                         f'with probability {second_prob}')

        st7, st8 = st.columns(2)
        start_time = 0
        end_time = 0
        with st7:
            start_time = st.number_input('Insert a start time (second)')
        with st8:
            end_time = st.number_input('Insert a end time (second)')
        if start_time >0 and end_time>0:
            button_3 = st.button("Get segment result")
            if button_3:
                largest_label, largest_prob, second_label, second_prob = get_inference_audio_segment(
                    classifier,
                    'audios/audio.wav', start_time, end_time)
                st.markdown('The entire music is predicted with labels as follows:')
                st.write(f'Predicted label is {largest_label} '
                         f'with probability {largest_prob}')
                st.write(f'The second possible genre could be {second_label} '
                         f'with probability {second_prob}')

