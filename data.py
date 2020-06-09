# Amie Roten
# CS559: Term Project
# Data Processing Module

import os
from pathlib import Path

from sklearn.model_selection import train_test_split
import python_speech_features
from scipy.io import wavfile
import numpy as np


emotion_map = {1: "neutral",
               2: "calm",
               3: "happy",
               4: "sad",
               5: "angry",
               6: "fearful",
               7: "disgust",
               8: "surprised"}


class Corpus:
    all = []

class Sentence:
    def __init__(self, emotion, wav, fs, intensity, actor, statement):
        self.emotion = emotion_map[emotion]
        self.emoclass = emotion-1
        self.wav = wav
        self.fs = fs
        self.intensity = intensity
        self.actor = actor
        self.statement = statement
        self.features = python_speech_features.base.fbank(wav, samplerate=fs)[0]

    def getX_y(self):
        emotion_y = np.array([self.emoclass for x in range(len(self.features))])
        return self.features, emotion_y

if __name__ == "__main__":
    data_path = Path("data", "speech-emotion-recognition-ravdess-data")
    all_actors = list(os.walk(data_path))[0][1]
    corpus = Corpus()

    for actor in all_actors:
        actor_path = data_path.joinpath(actor)
        all_sentences = list(os.walk(actor_path))[0][2]
        for sentence in all_sentences:
            file_path = actor_path.joinpath(sentence)
            fs, wav = wavfile.read(file_path)
            _, _, emotion, intensity, statement, _, actor = sentence.split("-")
            corpus.all.append(Sentence(int(emotion), wav, fs, intensity, actor.split(".")[0], statement))

    print("pause")



