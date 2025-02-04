from lesionSeg.Config.configuration import ConfigurationManager
from lesionSeg.Component.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        ingestion_config = config.data_ingestion_config()
        data_ingestion = DataIngestion(config = ingestion_config)
        data_ingestion.extract_dataset()