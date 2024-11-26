import os
import shutil
import sys
from tqdm import tqdm
import itertools
import pandas as pd
from shutil import copyfile
from dotenv import load_dotenv

load_dotenv()
data_root_folder = os.environ.get("ORIGIN_FOLDER")

save_dir = os.environ.get("DATASET_FOLDER")
csv = os.environ.get("ORIGIN_FOLDER") + "train.csv"
csv_file = pd.read_csv(csv)


class DataHandler:
    def __init__(self):
        self.data_root_folder = data_root_folder
        self.save_dir = save_dir
        self.csv_file = csv_file

    def create_folder(self):
        if not os.path.isdir(self.save_dir + "train_arranged"):
            os.mkdir(self.save_dir + "train_arranged")
            print("Folder created")
        else:
            print("Folder already exist")

    def clear_folder(self):
        folder = self.save_dir
        # remove all contents in folder
        for filename in tqdm(os.listdir(folder)):
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
        self.clear_folder()
        self.create_folder()

        with tqdm(total=len(self.csv_file)) as pbar:
            for i in range(len(self.csv_file)):
                if not (
                    os.path.isdir(
                        self.save_dir + "train_arranged/" + self.csv_file["label"][i]
                    )
                ):
                    os.mkdir(
                        self.save_dir + "train_arranged/" + self.csv_file["label"][i]
                    )
                    shutil.copyfile(
                        self.data_root_folder
                        + "audio_train/"
                        + self.csv_file["fname"][i],
                        self.save_dir
                        + "train_arranged/"
                        + self.csv_file["label"][i]
                        + "/"
                        + self.csv_file["fname"][i],
                    )
                elif os.path.isdir(
                    self.save_dir + "train_arranged/" + self.csv_file["label"][i]
                ):
                    shutil.copyfile(
                        self.data_root_folder
                        + "audio_train/"
                        + self.csv_file["fname"][i],
                        self.save_dir
                        + "train_arranged/"
                        + self.csv_file["label"][i]
                        + "/"
                        + self.csv_file["fname"][i],
                    )
                pbar.update(1)
        print("Data arranged")
