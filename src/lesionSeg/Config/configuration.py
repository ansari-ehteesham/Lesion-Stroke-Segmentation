from pathlib import Path
from lesionSeg.constant import *
from lesionSeg.Utils.common import read_yaml, create_directory
from lesionSeg.entity import (DataIngestionEntity,
                              DataValidationEntity,
                              DataProcessingEntity,
                              ModelTrainingEntity,
                              PredictionEntity)


class ConfigurationManager:
    def __init__(self, parmas_file = PARAMS_FILE_PATH, config_file = CONFIG_FILE_PATH):
        self.params = read_yaml(parmas_file)
        self.config = read_yaml(config_file)

        create_directory([self.config.artifact_root])

    def data_ingestion_config(self) -> DataIngestionEntity:
        config = self.config.data_ingestion
        secret = read_yaml(Path(config.secret_dir))

        data_ingestion = DataIngestionEntity(
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
            testing_data = Path(config.testing_data),
            preprocess_train_in = Path(config.preprocess_train_in),
            preprocess_train_op = Path(config.preprocess_train_op),
            preprocess_test_in = Path(config.preprocess_test_in),
            rm_data = Path(config.rm_data),
            img_height = params.img_height,
            img_width = params.img_width,
            slice_stride = params.slice_stride
        )
        
        return data_preprocessing_entity
    
    def model_training_config(self):
        config = self.config.model_training
        params = self.params.model_training

        create_directory([config.root_dir, config.trained_model])

        training_entity = ModelTrainingEntity(
            train_input_data = Path(config.train_input_data),
            train_mask_data = Path(config.train_mask_data),
            trained_model= Path(config.trained_model),
            tensorboard_logs = Path(config.tensorboard_logs),
            batch_size = params.batch_size,
            nfilters = params.nfilters,
            drop_out = params.drop_out,
            nclasses = params.nclasses,
            focal_alpha_loss = params.focal_alpha_loss,
            focal_gamma_loss = params.focal_gamma_loss,
            learning_rate = params.learning_rate,
            input_height = params.input_height,
            input_width = params.input_width,
            input_channel = params.input_channel,
            final_class_activation = params.final_class_activation,
            activation = params.activation,
            kernel_initializer = params.kernel_initializer,
            epochs = params.epochs,
        )

        return training_entity
    
    def prediction_config(self):
        config = self.config.prediction

        pred_config = PredictionEntity(
            model=config.base_model
        )

        return pred_config
