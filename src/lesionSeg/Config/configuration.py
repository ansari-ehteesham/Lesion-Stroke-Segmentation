from pathlib import Path
from lesionSeg.constant import *
from lesionSeg.Utils.common import read_yaml, create_directory
from lesionSeg.entity import (DataIngestionEntity,
                              DataValidationEntity)


class ConfigurationManager:
    def __init__(self, parmas_file = PARAMS_FILE_PATH, config_file = CONFIG_FILE_PATH):
        self.parmas = read_yaml(parmas_file)
        self.config = read_yaml(config_file)

        create_directory([self.config.artifact_root])

    def data_ingestion_config(self) -> DataIngestionEntity:
        config = self.config.data_ingestion
        secret = read_yaml(Path(config.secret_dir))

        data_ingestion = DataIngestionEntity(
            encrypted_dataset = Path(config.encrypted_dataset),
            zip_dataset = Path(config.zip_dataset),
            unzip_dir = Path(config.unzip_dir),
            password = secret.dataset_password,
        )

        return data_ingestion
    
    def data_validation_config(self):
        config = self.config.data_validation

        create_directory([config.sample_op_dir])

        data_validation = DataValidationEntity(
            root_dir = config.root_dir,
            animation_file = config.animation_file,
            report_file = config.report_file,
        )

        return data_validation
