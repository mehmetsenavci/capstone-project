import requests
import json
from predictWord import Keyword_Spotting_Service

# server url
URL = "http://127.0.0.1:5000/predict"


# audio file we'd like to send for predicting keyword
FILE_PATH = "Test_Dataset/four/a6285644_nohash_2.wav"


if __name__ == "__main__":

    # open files
    kss = Keyword_Spotting_Service()
    word = kss.predict(FILE_PATH) # Causes error!

    # package stuff to send and perform POST request
    # Edit down below when error fixed!
    values = {'word': word}
    requests.post(
        URL,
        json=values
    )
    
   
    # print("Predicted keyword: {}".format(data["keyword"]))