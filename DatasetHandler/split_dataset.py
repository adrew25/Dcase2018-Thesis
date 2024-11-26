import random
import os
import shutil
import dotenv
dotenv.load_dotenv()



def split_dataset():

    # from dataset_folder take a random 20% of all classes and move them to valid_set_folder
    dataset_folder = os.getenv("DATASET_FOLDER")
    valid_set_folder = os.getenv("DATASET_FOLDER") + "valid_set/"

    # get all classes
    classes = os.listdir(dataset_folder)
    print(classes)

    # create valid_set_folder if it doesn't exist
    if not os.path.exists(valid_set_folder):
        os.mkdir(valid_set_folder)

    # for each class in classes create a folder in valid_set_folder if it doesn't exist
    for class_name in classes:
        if not os.path.exists(valid_set_folder + class_name):
            os.mkdir(valid_set_folder + class_name)

    # for each class in classes get all files in dataset_folder/class_name and move 20% of them to valid_set_folder/class_name
    for class_name in classes:
        files = os.listdir(dataset_folder + class_name)
        files_to_move = random.sample(files, int(len(files) * 0.2))
        for file in files_to_move:
            shutil.move(dataset_folder + class_name + "/" + file, valid_set_folder + class_name + "/" + file)