from pathlib import Path
from lesionSeg.constant import *
from lesionSeg.Utils.common import read_yaml, create_directory
from lesionSeg.entity import (DataIngestionEntity,
                              DataValidationEntity,
                              DataProcessingEntity)


class ConfigurationManager:
    def __init__(self, parmas_file = PARAMS_FILE_PATH, config_file = CONFIG_FILE_PATH):
        self.params = read_yaml(parmas_file)
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
            root_dir = Path(config.root_dir),
            animation_file = Path(config.animation_file),
            report_file = Path(config.report_file),
        )

        return data_validation
    
    def data_preprocessing_config(self) -> DataProcessingEntity:
        params = self.params.image_preprocess
        config = self.config.data_preprocessing

        create_directory([config.root_dir, 
                          config.preprocess_train_in, 
                          config.preprocess_train_op, 
                          config.preprocess_test_in])

        data_preprocessing_entity = DataProcessingEntity(
            training_data = Path(config.training_data),
            train_csv = Path(config.train_csv),
            test_csv = Path(config.test_csv),
            testing_data = Path(config.testing_data),
            preprocess_train_in = Path(config.preprocess_train_in),
            preprocess_train_op = Path(config.preprocess_train_op),
            preprocess_test_in = Path(config.preprocess_test_in),
            img_height = params.img_height,
            img_width = params.img_width,
            img_norms_range = params.img_norms_range
        )
        
        return data_preprocessing_entity
