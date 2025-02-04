import sys, os
import shutil
import tarfile 
import subprocess
from lesionSeg.logging import logger
from lesionSeg.Exception.exception import CustomeException
from lesionSeg.entity import DataIngestionEntity



class DataIngestion:
    def __init__(self, config: DataIngestionEntity):
        self.config = config
        
    def rename_folder(self, unzip_dir):
        for file in os.listdir(unzip_dir):   
            if "ATLAS" in file:
                os.rename(f"{unzip_dir}/{file}", f"{unzip_dir}/data_ingestion")
                logger.info(f"File Name Changed from {file} to data_ingestion")
                break

    def decrypt_dataset(self, zip_file):
        file_name = self.config.encrypted_dataset
        try:
            if not shutil.which('openssl'):
                message = "OpenSSL is not Installed. Please Installed OpenSSL"
                raise ModuleNotFoundError(message)
            else:
                password = self.config.password
                cmd = [
                    "openssl",
                    "enc",            
                    "-aes-256-cbc",
                    "-md", "sha256",
                    "-d",             
                    "-a",             
                    "-in", file_name,
                    "-out", zip_file,
                    "-pass", f"pass:{password}"
                ]

                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                logger.info(f"Dataset Decrypted Successfully at: {zip_file}")

        except Exception as e:
            raise CustomeException(e,sys)
        

    def extract_dataset(self):
        unzip_dir = self.config.unzip_dir
        zip_file = self.config.zip_dataset
        self.decrypt_dataset(unzip_dir, zip_file)

        # Unzip File
        try:
            file = tarfile.open(zip_file)
            file.extractall(unzip_dir) 
            file.close()
            logger.info(f"Dataset Unzipped at: {unzip_dir}/{zip_file}")
        except Exception as e:
            raise CustomeException(e,sys)
        
        self.rename_folder(unzip_dir)
        