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