import tensorflow as tf
from sklearn.model_selection import train_test_split
from lesionSeg.models.unet_2d import UNet2D
from lesionSeg.logging import logger
from lesionSeg.Utils.common import CustomDataGenerator
from lesionSeg.Exception.exception import CustomeException
from lesionSeg.entity import ModelTrainingEntity
from lesionSeg.models.loss import focal_tversky_loss
from lesionSeg.models.metrices import dice_coff
import os
from args import get_args
from typing import Tuple
from pathlib import Path

import numpy as np
np.Inf = np.inf


class ModelTraining:
    def __init__(self, config: ModelTrainingEntity):
        self.args = get_args()
        self.config = config
        self.input_shape = (self.config.input_height, 
                          self.config.input_width, 
                          self.config.input_channel)
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validate existence of required directories"""
        
        required_paths = {
            'input_dir': self.config.train_input_data,
            'output_dir': self.config.train_mask_data,
        }
        
        if self.args.train_type == 'fine_tune':
            if self.args.model_path != None:
                required_paths['model_path'] = self.args.model_path

        for name, path in required_paths.items():
            if not Path(path).exists():
                raise CustomeException(f"{name} path does not exist: {path}")

    def _create_data_generators(self) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
        """Create training and validation data generators"""
        
        input_dir = self.config.train_input_data
        output_dir = self.config.train_mask_data
        # Get sorted file lists with validation
        input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        output_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)])

        # Verify file correspondence
        if len(input_files) != len(output_files):
            raise CustomeException("Mismatch between input and output file counts")
            
        # Split data
        X_train, X_valid, y_train, y_valid = train_test_split(
            input_files, output_files, 
            test_size=0.1, 
            random_state=42
        )
        logger.info(f"Training Samples: {len(X_train)}, Validation Samples: {len(X_valid)}")

        # Create generators
        train_gen = CustomDataGenerator(
            input_lst=X_train,
            output_lst=y_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            augmentation=True
        )
        
        valid_gen = CustomDataGenerator(
            input_lst=X_valid,
            output_lst=y_valid,
            batch_size=self.config.batch_size,
            shuffle=False,
            augmentation=False
        )
        
        return train_gen, valid_gen

    def _get_model_callbacks(self) -> list:
        """Create and return training callbacks"""
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.config.trained_model, "model-{epoch:03d}-{val_loss:.4f}.keras"),
            monitor="val_iou",
            mode="max",
            save_best_only=True,
            verbose=1,
            save_weights_only=False,
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_iou",
            patience=15,
            mode="max",
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=self.config.tensorboard_logs
        )

        return [checkpoint, early_stopping, reduce_lr, tensorboard]
    
    
    def _initialize_model(self) -> tf.keras.models:
        """Initialize or load model based on training type"""
        if self.args.train_type == 'new':
            logger.info("Initializing new model")
            model = UNet2D(
                nfilters=self.config.nfilters,
                nclassess=self.config.nclasses,
                final_class_activation=self.config.final_class_activation,
                activation=self.config.activation,
                kernel_initializer=self.config.kernel_initializer,
                input_size=self.input_shape
            )
            unet = model.unet_model()
            return unet
            
        elif self.args.train_type == 'fine_tune':
            if self.args.model_path != None:
                logger.info(f"Loading pre-trained model from {self.args.model_path}")
        
                model_dir = self.args.model_path
                
            else:
                model_dir = "best_model.keras"
                logger.info(f"Default Model is Loading: {model_dir}")
            
            try:
                model = tf.keras.models.load_model(
                        model_dir,
                        custom_objects = {
                            'focal_tversky_loss': focal_tversky_loss,
                            'dice_coff': dice_coff
                        },
                        compile = False
                    )
                return model
            except Exception as e:
                raise CustomeException(f"Error Loading Model: {str(e)}")

    def train_model(self) -> None:
        """Main training workflow"""
        
        # Create data generators
        train_gen, valid_gen = self._create_data_generators()
        
        # Initialize model
        model = self._initialize_model()
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=focal_tversky_loss,
            metrics=[
                dice_coff,
                tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1], name='iou'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        logger.info("Model Summary")
        print(model.summary())

        if self.args.epochs:
            epochs = self.args.epochs
        else:
            epochs = self.config.epochs
        
        # Train model
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=valid_gen,
            callbacks=self._get_model_callbacks(),
            verbose=1
        )
        
        logger.info("Training completed successfully")