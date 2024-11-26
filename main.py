from DatasetHandler.DataHandler import DataHandler
from DatasetHandler.audio_image import StartMain, TestSpecs

data_handler = DataHandler()
audio_image = StartMain()
audio_test = TestSpecs()

if __name__ == "__main__":
    # for Audio train dataset
    # data_handler.arrange_from_csv()

    # for npy train dataset
    audio_image
    audio_test
