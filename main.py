from lesionSeg.logging import logger
from lesionSeg.Pipeline.stage_01 import DataIngestionTrainingPipeline



STAGE_NAME = "Data Ingestion Stage"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_ingestion = DataIngestionTrainingPipeline()
data_ingestion.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")
