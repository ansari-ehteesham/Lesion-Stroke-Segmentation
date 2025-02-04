from lesionSeg.Config.configuration import ConfiguratioManager
from lesionSeg.Component.data_ingestion import DataIngestion

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfiguratioManager()
        ingestion_config = config.data_ingestion_config()
        data_ingestion = DataIngestion(config = ingestion_config)
        data_ingestion.extract_dataset()