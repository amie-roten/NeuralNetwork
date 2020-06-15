# Amie Roten
# CS559: Term Project
# Data Processing Module

import os
from pathlib import Path

from sklearn.model_selection import train_test_split
import python_speech_features
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import audiosegment


emotion_map = {1: "neutral",
               2: "calm",
               3: "happy",
               4: "sad",
               5: "angry",
               6: "fearful",
               7: "disgust",
               8: "surprised"}

emotion_map_find_class = {"neutral": 0,
                          "calm": 1,
                          "happy": 2,
                          "sad": 3,
                          "angry": 4,
                          "fearful": 5,
                          "disgust": 6,
                          "surprised": 7}

class Sentence:
    def __init__(self, emotion, wav, fs, intensity, actor, statement):
        self.emotion = emotion_map[emotion]
        self.emoclass = emotion-1
        self.wav = wav
        self.fs = fs
        self.intensity = intensity
        self.actor = int(actor)
        self.statement = statement
        self.features = python_speech_features.base.fbank(wav, samplerate=fs)[0]

    def get_X(self):
        return np.mean(self.features, axis=0)

    def get_y(self):
        return self.emoclass

    # Odd actor numbers are male,
    # even are female. So, if we
    # want to divide the class by
    # gender, the first 8 classes
    # are male speakers, and the
    # second 8 are female.
    def get_y_gendered(self):
        if self.actor % 2 == 0:
            return self.emoclass + 8
        return self.emoclass


class Corpus:
    all_sentences = []

    def __init__(self):
        skipped = 0
        #data_path = Path("data", "speech-emotion-recognition-ravdess-data")
        data_path = Path("data", "silence_removed")
        all_actors = list(os.walk(data_path))[0][1]

        for actor in all_actors:
            actor_path = data_path.joinpath(actor)
            all_sentences = list(os.walk(actor_path))[0][2]
            for sentence in all_sentences:
                file_path = actor_path.joinpath(sentence)
                fs, wav = wavfile.read(file_path)
                if len(wav) == 0:
                    print("fileskipped")
                    skipped +=1
                    continue
                _, _, emotion, intensity, statement, _, actor = sentence.split("-")
                self.all_sentences.append(Sentence(int(emotion), wav, fs, intensity, actor.split(".")[0], statement))
        print("Total skipped:", skipped, " out of", len(self.all_sentences))

    def get_all_data(self):
        X = np.array(self.all_sentences[0].get_X())
        y = [self.all_sentences[0].get_y()]
        for sentence in self.all_sentences[1:]:
            X = np.vstack((X, sentence.get_X()))
            y.append(sentence.get_y())
        return X, y

    def get_all_data_gendered(self):
        X = np.array(self.all_sentences[0].get_X())
        y = [self.all_sentences[0].get_y_gendered()]
        for sentence in self.all_sentences[1:]:
            X = np.vstack((X, sentence.get_X()))
            y.append(sentence.get_y_gendered())
        return X, y

def strip_silence_all():
    data_path = Path("data", "speech-emotion-recognition-ravdess-data")
    all_actors = list(os.walk(data_path))[0][1]

    for actor in all_actors:
        actor_path = data_path.joinpath(actor)
        all_sentences = list(os.walk(actor_path))[0][2]

        try:
            os.mkdir(Path("data", "silence_removed", actor))
        except FileExistsError:
            pass

        for sentence in all_sentences:
            file_path = actor_path.joinpath(sentence)
            pre_stripped = audiosegment.from_file(file_path)
            post_stripped = pre_stripped.filter_silence(duration_s=0.75,
                                                        threshold_percentage=0.2)
            new_path = Path("data","silence_removed", actor, sentence)
            post_stripped.export(new_path, format="wav")


if __name__ == "__main__":

    corpus = Corpus()

    # This really only needs to be done once, but
    # keeping the function around to perhaps be
    # used in the future.
    #strip_silence_all()

    # data_path = Path("data", "silence_removed")
    # all_actors = list(os.walk(data_path))[0][1]
    #
    # for actor in all_actors:
    #     actor_path = data_path.joinpath(actor)
    #     all_sentences = list(os.walk(actor_path))[0][2]
    #     for sentence in all_sentences:
    #         file_path = actor_path.joinpath(sentence)
    #         fs, wav = wavfile.read(file_path)
    #         _, _, emotion, intensity, statement, _, actor = sentence.split("-")
    #         corpus.all_sentences.append(Sentence(int(emotion), wav, fs, intensity, actor.split(".")[0], statement))

    X, y = corpus.get_all_data()
    print("pause")



