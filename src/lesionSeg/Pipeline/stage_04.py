from lesionSeg.Config.configuration import ConfigurationManager
from lesionSeg.Component.model_training import ModelTraining


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.model_training_config()
        model_training = ModelTraining(training_config)
        model_training.train_model()