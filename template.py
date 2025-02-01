import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

PROJECT_NAME = "lesionSeg"

files_lst = [
    "app.py",       # Web Application 
    "main.py",      
    "setup.py",     # Help to Install your Project as Package
    "params.yaml",  # Storing Hyper-parameter and model 
    "Dockerfile",   # Setup Docker Images
    "requirements.txt",     # Lists all dependencies              
    "config/config.yaml",   # Store directories and general application related settings
    "Documents/Introduction.docx",      # Documents of Project
    "notebooks/trial.ipynb",    # Jupyter Notebook Experiment
    f"src/{PROJECT_NAME}/__init__.py",      # Python Package Initialiased
    
    f"src/{PROJECT_NAME}/Config/__init__.py",      # Central Manager for managing and 
    f"src/{PROJECT_NAME}/Config/configuration.py", # organising project settings     
    
    f"src/{PROJECT_NAME}/logging/__init__.py",      # Custom Logger
    
    f"src/{PROJECT_NAME}/Component/__init__.py",    # Smaller Units of Model Development
    
    f"src/{PROJECT_NAME}/Pipeline/__init__.py",     # Pipeline for Executing Model Components
    
    f"src/{PROJECT_NAME}/constant/__init__.py",     # Constant variables
    
    f"src/{PROJECT_NAME}/Utils/__init__.py",        # Utility which are used frequently
    f"src/{PROJECT_NAME}/Utils/common.py",      
    
    f"src/{PROJECT_NAME}/entity/__init__.py",      # Defines dtype of Input of Model Components

    f"src/{PROJECT_NAME}/Exception/__init__.py",   
    f"src/{PROJECT_NAME}/Exception/exception.py",   # Custome Exception Handling

]

for file in files_lst:
    file = Path(file)
    file_dir, file_name = os.path.split(file)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Directory is Created: {file_dir}")
    
    if (not os.path.exists(file)) or (os.path.getsize(file)==0):
        with open(file, 'w') as f:
            pass
        logging.info(f"Files is Created: {file}")
    else:
        logging.info(f"File Alread Exists: {file}")