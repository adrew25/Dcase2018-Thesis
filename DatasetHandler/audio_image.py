import os
import shutil
import pickle
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

from dotenv import load_dotenv

load_dotenv()


class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(
            file_path, sr=self.sample_rate, duration=self.duration, mono=self.mono
        )[0]
        return signal


class Padder:
    """Padder is responsible to apply padding to an array."""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[
            :-1
        ]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        log_spectrogram = np.flipud(log_spectrogram)
        return log_spectrogram


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class Saver:
    """saver is responsible to save features, and the min max values."""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        # np.save(save_path, feature)
        cv2.imwrite(save_path, feature * 255)

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".png")
        return save_path


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal
        4- normalise spectrogram
        5- save the normalised spectrogram
    Storing the min max values for all the log spectrograms.
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        counter_i = 0
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                counter_i = counter_i + 1
                print(f"Processed file {file_path} ---> {counter_i}", end="\r")
        print()
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {"min": min_val, "max": max_val}


class DataArranger:
    def __init__(self):
        self.data_root_folder = os.environ.get("ORIGIN_FOLDER")
        self.save_dir = os.environ.get("DATASET_FOLDER")
        self.csv_file = pd.read_csv(os.environ.get("ORIGIN_FOLDER") + "train.csv")

    def create_folder(self):
        if not os.path.isdir(self.save_dir + "specs_arranged"):
            os.mkdir(self.save_dir + "specs_arranged")
            print("Folder created")
        else:
            print("Folder already exist")

    def clear_folder(self):
        folder = self.save_dir + "specs_arranged"
        # remove all contents in folder
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
        print("Folder cleared")

    def arrange_from_csv(self):
        self.create_folder()
        self.clear_folder()

        with tqdm(total=len(self.csv_file)) as pbar:
            for i in range(len(self.csv_file)):
                if not (
                    os.path.isdir(
                        self.save_dir + "specs_arranged/" + self.csv_file["label"][i]
                    )
                ):
                    os.mkdir(
                        self.save_dir + "specs_arranged/" + self.csv_file["label"][i]
                    )
                    try:
                        shutil.copyfile(
                            self.save_dir
                            + "spectrograms/"
                            + self.csv_file["fname"][i]
                            + ".png",
                            self.save_dir
                            + "specs_arranged/"
                            + self.csv_file["label"][i]
                            + "/"
                            + self.csv_file["fname"][i]
                            + ".png",
                        )
                    except:
                        print("File not found: ", self.csv_file["fname"][i] + ".png")

                elif os.path.isdir(
                    self.save_dir + "specs_arranged/" + self.csv_file["label"][i]
                ):
                    try:
                        shutil.copyfile(
                            self.save_dir
                            + "spectrograms/"
                            + self.csv_file["fname"][i]
                            + ".png",
                            self.save_dir
                            + "specs_arranged/"
                            + self.csv_file["label"][i]
                            + "/"
                            + self.csv_file["fname"][i]
                            + ".png",
                        )
                    except:
                        print("File not found: ", self.csv_file["fname"][i] + ".png")
                pbar.update(1)
        count_in_specs = 0
        for root, _, files in os.walk(self.save_dir + "specs_arranged/"):
            for file in files:
                count_in_specs = count_in_specs + 1
        print("Files in specs_arranged: ", count_in_specs)


class TestSpecs:
    FRAME_SIZE = 1024
    HOP_LENGTH = 256
    DURATION = 10  # in seconds
    SAMPLE_RATE = 44100
    MONO = True

    if not os.path.isdir(os.environ.get("DATASET_FOLDER") + "test_specs/"):
        os.mkdir(os.environ.get("DATASET_FOLDER") + "test_specs/")
        print("Folder created")
    else:
        print("Folder already exist")
    if not (os.path.isdir(os.environ.get("DATASET_FOLDER") + "test_min_max_values/")):
        os.mkdir(os.environ.get("DATASET_FOLDER") + "test_min_max_values/")
        print("Folder created")
    else:
        print("Folder already exist")

    SPECTROGRAMS_SAVE_DIR = os.environ.get("DATASET_FOLDER") + "test_specs/"
    MIN_MAX_VALUES_SAVE_DIR = os.environ.get("DATASET_FOLDER") + "test_min_max_values/"
    FILES_DIR_TEST = os.environ.get("ORIGIN_FOLDER") + "audio_test/"

    count_in_train = 0
    for root, _, files in os.walk(FILES_DIR_TEST):
        for file in files:
            count_in_train = count_in_train + 1
    print("Files in test_specs: ", count_in_train)

    count_in_specs = 0
    for root, _, files in os.walk(SPECTROGRAMS_SAVE_DIR):
        for file in files:
            count_in_specs = count_in_specs + 1
    print("Files in audio_test: ", count_in_specs)

    count_in_train = count_in_train - 3
    print("Files in test_specs - 3: ", count_in_specs)

    if count_in_train == count_in_specs:
        print("Preprocessing skipped")
        exit()
    else:
        print("Preprocessing started")
        # initialise all objects
        loader = Loader(SAMPLE_RATE, DURATION, MONO)
        padder = Padder()
        log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
        min_max_normaliser = MinMaxNormaliser(0, 1)
        saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

        preprocessing_pipeline = PreprocessingPipeline()
        preprocessing_pipeline.loader = loader
        preprocessing_pipeline.padder = padder
        preprocessing_pipeline.extractor = log_spectrogram_extractor
        preprocessing_pipeline.normaliser = min_max_normaliser
        preprocessing_pipeline.saver = saver
        preprocessing_pipeline.process(FILES_DIR_TEST)
        print("Preprocessing finished")

        bad_list = ["6ea0099f.wav", "b39975f5.wav", "0b0427e2.wav"]
        for i in bad_list:
            os.remove(SPECTROGRAMS_SAVE_DIR + i + ".png")
            print("Removed: ", i)
        print("Go train your model now!")


class StartMain:
    FRAME_SIZE = 1024
    HOP_LENGTH = 256
    DURATION = 10  # in seconds
    SAMPLE_RATE = 44100
    MONO = True

    if not os.path.isdir(os.environ.get("DATASET_FOLDER") + "spectrograms/"):
        os.mkdir(os.environ.get("DATASET_FOLDER") + "spectrograms/")
        print("Folder created")
    else:
        print("Folder already exist")

    if not os.path.isdir(os.environ.get("DATASET_FOLDER") + "min_max_values/"):
        os.mkdir(os.environ.get("DATASET_FOLDER") + "min_max_values/")
        print("Folder created")
    else:
        print("Folder already exist")

    SPECTROGRAMS_SAVE_DIR = os.environ.get("DATASET_FOLDER") + "spectrograms/"
    MIN_MAX_VALUES_SAVE_DIR = os.environ.get("DATASET_FOLDER") + "min_max_values/"
    FILES_DIR_TRAIN = os.environ.get("DATASET_FOLDER") + "train_arranged/"

    count_in_train = 0
    for root, _, files in os.walk(FILES_DIR_TRAIN):
        for file in files:
            count_in_train = count_in_train + 1
    print("Files in train_arranged: ", count_in_train)

    count_in_specs = 0
    for root, _, files in os.walk(SPECTROGRAMS_SAVE_DIR):
        for file in files:
            count_in_specs = count_in_specs + 1
    print("Files in spectrograms: ", count_in_specs)

    if count_in_train == count_in_specs:
        print("Preprocessing skipped")
        data_arranger = DataArranger()
        data_arranger.arrange_from_csv()
        print("Data arranged \nGo train your model now!")
    else:
        print("Preprocessing started")
        # initialise all objects
        loader = Loader(SAMPLE_RATE, DURATION, MONO)
        padder = Padder()
        log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
        min_max_normaliser = MinMaxNormaliser(0, 1)
        saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

        preprocessing_pipeline = PreprocessingPipeline()
        preprocessing_pipeline.loader = loader
        preprocessing_pipeline.padder = padder
        preprocessing_pipeline.extractor = log_spectrogram_extractor
        preprocessing_pipeline.normaliser = min_max_normaliser
        preprocessing_pipeline.saver = saver
        preprocessing_pipeline.process(FILES_DIR_TRAIN)

        data_arranger = DataArranger()
        data_arranger.arrange_from_csv()
        print("Go train your model now!")
