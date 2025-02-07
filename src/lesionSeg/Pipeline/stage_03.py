from lesionSeg.Config.configuration import ConfigurationManager
from lesionSeg.Component.data_preprocessing import DataPreprocessing

class DataPreProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_processing_config = config.data_preprocessing_config()
        data_preprocessing = DataPreprocessing(data_processing_config)
        data_preprocessing.preprocess_dataset()