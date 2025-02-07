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
    img_norms_range: list