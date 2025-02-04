from lesionSeg.Config.configuration import ConfigurationManager
from lesionSeg.Component.data_validation import DataValidation


class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.data_validation_config()
        data_validation = DataValidation(data_validation_config)
        data_validation.load_dataset()