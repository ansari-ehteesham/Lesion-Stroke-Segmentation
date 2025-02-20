from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class DataIngestionEntity:
    encrypted_dataset: Path
    zip_dataset: Path
    unzip_dir: Path
    password: str


@dataclass(frozen=True)
class DataValidationEntity:
    root_dir: Path
    animation_file: Path
    report_file: Path


@dataclass(frozen=True)
class DataProcessingEntity:
    training_data: Path
    testing_data: Path
    train_csv: Path
    test_csv: Path
    preprocess_train_in: Path
    preprocess_train_op: Path
    preprocess_test_in: Path
    img_height: int
    img_width: int
    slice_stride: int

@dataclass(frozen=True)
class ModelTrainingEntity:
    train_input_data: Path
    train_mask_data: Path
    trained_model: Path
    tensorboard_logs: Path
    batch_size: int
    nfilters: int
    drop_out: float
    nclasses: int
    focal_alpha_loss: float
    focal_gamma_loss: float
    learning_rate: float
    input_height: int
    input_width: int
    input_channel: int
    epochs: int
    final_class_activation: str
    activation: str
    kernel_initializer: str