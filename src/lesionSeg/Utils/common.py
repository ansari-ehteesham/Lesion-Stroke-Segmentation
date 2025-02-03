import sys,os,yaml
from pathlib import Path
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from lesionSeg.logging import logger
from lesionSeg.Exception.exception import CustomeException, catch_ensure_errors



# Read YAML Files
@catch_ensure_errors
@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    """
    It Reads the YAML File

    Arguments:
        file_path: YAML file path 
    
    Return:
        Data: YAML File Data as ConfigBox
    """
    try:
        with open(file=file_path) as yaml_file:
            data = yaml.safe_load(yaml_file)
            logger.info(f"Read YAML File: {file_path}")
            return ConfigBox(data)
    except BoxValueError:
        raise ValueError("YAML FIle is EMPTY")
    except Exception as e:
        raise CustomeException(e, sys)
    

# Create Directories
@catch_ensure_errors
@ensure_annotations
def create_directory(lst_dir:list):
    """
    It create directory
    Arguments:
        lst_dir: list of directory

    Return:
        None
    """
    for dir in lst_dir:
        dir = Path(dir)
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Directory has been Created: {dir}")