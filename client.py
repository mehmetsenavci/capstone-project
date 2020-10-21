import requests
import json
from predictWord import Keyword_Spotting_Service
import wave
import pyaudio
from scipy.io.wavfile import read
import trimAudio
from soundfile import SoundFile, SEEK_END
import numpy as np

# server url
URL = "http://127.0.0.1:5000/predict"


# audio file we'd like to send for predicting keyword
FILE_PATH = "Test_Dataset/four/a6285644_nohash_2.wav"

# Constants for voice recording
RECORDED_AUDIO = "rawAudio.wav"
chunk = 1024
FORMAT = pyaudio.paInt16
channels = 1
sample_rate = 48000 #110250
record_seconds = 5
p = pyaudio.PyAudio()

if __name__ == "__main__":
    # 5 seconds voice recording
    stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)

    frames = []
    print("Recording...")
    for i in range(int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(RECORDED_AUDIO, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()


    # Trims raw audio file and saves as trimedAudio.wav
    trimAudio.trimAudio(1, 'rawAudio.wav')

   
    # Predicts word from the recorded wav file
    kss = Keyword_Spotting_Service()
    word = kss.predict('trimedAudio.wav')

   # Sends post request to the server
    values = {'word': word}
    requests.post(
        URL,
        json=values
    )
"""
    file = SoundFile('trimedAudio.wav', 'r+')
    end = file.seek(0, SEEK_END)
    file.write(np.array([2000]))
    print(end)
    file.close()
"""