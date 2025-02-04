from lesionSeg.logging import logger
from lesionSeg.Pipeline.stage_01 import DataIngestionPipeline
from lesionSeg.Pipeline.stage_02 import DataValidationPipeline



STAGE_NAME = "Data Ingestion Stage"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_ingestion = DataIngestionPipeline()
data_ingestion.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")


STAGE_NAME = "Data Validation Stage"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_ingestion = DataValidationPipeline()
data_ingestion.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")
