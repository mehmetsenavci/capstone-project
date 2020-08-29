import librosa
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


DATASET_PATH = "/content/drive/My Drive/Dataset"
JSON_PATH = "/content/drive/My Drive/data.json"
SAVED_MODEL_PATH = "/content/drive/My Drive/model.h5"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
NUM_KEYWORDS = 5
SAMPLES_TO_CONSIDER = 22050 # 1 sec worth of sound

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
  # data dict
  data = {
      "mappings": [],
      "labels": [],
      "MFCCs": [],
      "files": []
  }

  # loop through all the sub-directories
  for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    # we need to ensure that we are not at root level
    if dirpath is not dataset_path:
      category = dirpath.split("/")[-1]
      data["mappings"].append(category)

      print(f"Processing {category}")

      for f in filenames:
        file_path = os.path.join(dirpath, f)
        signal, sr = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
          signal = signal[:SAMPLES_TO_CONSIDER]

          MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

          data["labels"].append(i-1)
          data["MFCCs"].append(MFCCs.T.tolist())
          data["files"].append(file_path)

          print(f"{file_path}:{i-1}")

  with open(json_path, "w") as fp:
    json.dump(data, fp, indent=4)

if __name__ == "__main__":
  prepare_dataset(DATASET_PATH, JSON_PATH)



