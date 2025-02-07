from lesionSeg.logging import logger
from lesionSeg.Pipeline.stage_01 import DataIngestionPipeline
from lesionSeg.Pipeline.stage_02 import DataValidationPipeline
from lesionSeg.Pipeline.stage_03 import DataPreProcessingPipeline



STAGE_NAME = "Data Ingestion Stage"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_ingestion = DataIngestionPipeline()
data_ingestion.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")


STAGE_NAME = "Data Validation Stage"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_validation = DataValidationPipeline()
data_validation.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")



STAGE_NAME = "Data Pre-Processing Stage"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_preprocess = DataPreProcessingPipeline()
data_preprocess.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")
