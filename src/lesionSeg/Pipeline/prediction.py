from lesionSeg.Component.data_preprocessing import DataPreprocessing
from lesionSeg.models.loss import focal_tversky_loss
from lesionSeg.models.metrices import dice_coff
from lesionSeg.Config.configuration import ConfigurationManager
from lesionSeg.entity import PredictionEntity
import numpy as np
from scipy import ndimage
import tensorflow as tf
import cv2

class Prediction:
    def __init__(self):
        config_man = ConfigurationManager()
        self.config = config_man.prediction_config()
        preprocess_config = config_man.data_preprocessing_config()

        self.pre_process_config = DataPreprocessing(config=preprocess_config)

    def img_preprocess(self, image):
        # resize the image
        if image.ndim < 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height_factor, width_factor = self.pre_process_config.calculate_resize_factors(image)
        resized_image = ndimage.zoom(image, (width_factor, height_factor, 1), order=1)
        img = np.expand_dims(resized_image, axis=0)

        return img

    def model_prediction(self, image):
        # load model
        model_dir = self.config.model
        model = tf.keras.models.load_model(model_dir, 
                                           custom_objects={
                        'focal_tversky_loss': focal_tversky_loss,
                        'dice_coff': dice_coff,
                        'iou':tf.keras.metrics.IoU,
                        'precision':tf.keras.metrics.Precision,
                        'recall':tf.keras.metrics.Recall
                    })
        pre_process_img = self.img_preprocess(image=image)
        pred = model.predict(pre_process_img)
        pred_mask = (pred > 0.5).astype(np.uint8)

        return pred_mask[0]