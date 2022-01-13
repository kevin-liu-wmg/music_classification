from inference_package.inference import ClassificationEngine

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
          'pop', 'reggae', 'rock']
audio_root_path = '/Users/zhanlinliu/Desktop/audio_projects/music_classification/data/genres/'
model_type = "Transfer_Cnn14"
model_path = "/Users/zhanlinliu/Desktop/audio_projects/music_classification/panns_transfer_to_gtzan/checkpoints/BestAcc.pth"
device = -1
classifier = ClassificationEngine(model_type,
                                  model_path, device)