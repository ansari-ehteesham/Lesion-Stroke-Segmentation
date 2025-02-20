from args import get_args
from lesionSeg.logging import logger
from lesionSeg.Pipeline.stage_01 import DataIngestionPipeline
from lesionSeg.Pipeline.stage_02 import DataValidationPipeline
from lesionSeg.Pipeline.stage_03 import DataPreProcessingPipeline
from lesionSeg.Pipeline.stage_04 import ModelTrainingPipeline




STAGE_NAME = "Data Ingestion"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_ingestion = DataIngestionPipeline()
data_ingestion.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")


STAGE_NAME = "Data Validation"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_validation = DataValidationPipeline()
data_validation.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")



STAGE_NAME = "Data Pre-Processing"

logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
data_preprocess = DataPreProcessingPipeline()
data_preprocess.main()
logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")


args = get_args()
if args.mode == 'train':
    STAGE_NAME = "Model Training"

    logger.info(f">>>>>> STAGE {STAGE_NAME} STARTED <<<<<<") 
    model_training = ModelTrainingPipeline()
    model_training.main()
    logger.info(f">>>>>> STAGE {STAGE_NAME} COMPLETED <<<<<<\n\nx==========x")

elif args.mode == 'pretrained':
    print(F"{'-'*20} COMING SOON {'-'*20}")