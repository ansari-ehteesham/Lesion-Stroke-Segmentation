import os
import sys
import logging

logging_format = "[%(asctime)s]: %(levelname)s: %(module)s : %(message)s"
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    format=logging_format,
    level=logging.INFO,

    handlers=[
        logging.FileHandler(filename=log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("lesionSegmentationlogger")


logging.getLogger("nibabel").setLevel(logging.ERROR)
logging.getLogger("matplotlib.animation").setLevel(logging.ERROR)
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)