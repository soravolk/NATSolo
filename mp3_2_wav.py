import os
from pydub import AudioSegment

# files                                                                         
path = "./Solo/train_unlabel/wav"

filenames = os.listdir(path)
for filename in filenames:
    # convert wav to mp3                                                            
    file = f"{path}/{filename}"
    print(file)
    sound = AudioSegment.from_mp3(file)
    sound.export(f"{file[:-4]}.wav", format="wav")
    os.remove(file)
