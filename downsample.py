import os
from glob import glob
from pydub import AudioSegment

# files                                                                         
# path = glob('./Solo/*/wav/*.wav')
# print(path)
folders = glob('./Solo/*/wav')
for folder in folders:
    for wavfile in glob(f'{folder}/*.wav'):
        filename = wavfile.split('/')[-1]
        sound = AudioSegment.from_wav(wavfile)
        sound = sound.set_frame_rate(16000) # downsample it to 16000
        sound = sound.set_channels(1) # Convert Stereo to Mono
        sound.export(folder[:-3] + 'flac/' + filename[:-3] + 'flac', format='flac')

# for wavfile in glob('./Solo/*/wav/*.wav'):
#     filename = wavfile.split('/')[-1]
#     print(wavfile.split('/')[-1])
#     sound = AudioSegment.from_wav(wavfile)
#     sound = sound.set_frame_rate(16000) # downsample it to 16000
#     sound = sound.set_channels(1) # Convert Stereo to Mono
    
#     sound.export( + filename[:-3] + 'flac', format='flac')