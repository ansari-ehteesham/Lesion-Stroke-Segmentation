import os
import cv2
import sys
import pandas as pd
import numpy as np
import nibabel as nb
from bids import BIDSLayout
from lesionSeg.logging import logger
from lesionSeg.Exception.exception import CustomeException
from lesionSeg.entity import DataProcessingEntity
from scipy import ndimage
from typing import Dict, Tuple
import logging

logging.getLogger("nibabel").setLevel(logging.ERROR)



class DataPreprocessing:
    def __init__(self, config: DataProcessingEntity):
        self.config = config

    def calculate_resize_factors(self, slice_data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate resize factors for the given slice data.
        """
        current_width, current_height = slice_data.shape[:2]
        DESIRED_HEIGHT = self.config.img_height
        DESIRED_WIDTH = self.config.img_width
        height_factor = DESIRED_HEIGHT / current_height
        width_factor = DESIRED_WIDTH / current_width
        return height_factor, width_factor

    def train_df_process(self, df) -> pd.DataFrame:
        """
        Process the training dataframe by selecting the necessary columns and merging input and mask files.
        """
        selected_cols = ["path", "subject"]
        df_input = df[df['suffix'] == 'T1w'][selected_cols].rename(columns={'path': 'og_input_path'})
        df_mask = df[df['suffix'] == 'mask'][selected_cols].rename(columns={'path': 'og_mask_path'})
        df_merge = df_input.merge(df_mask, on='subject', how='inner').reset_index(drop=True)
        return df_merge[['og_input_path', 'og_mask_path', 'subject']]

    def test_df_process(self, df) -> pd.DataFrame:
        """
        Process the testing dataframe by selecting the necessary columns.
        """
        selected_cols = ["path", "subject"]
        df_input = df[df['suffix'] == 'T1w'][selected_cols].rename(columns={'path': 'og_input_path'})
        return df_input

    def _process_slice(self, slice_data: np.ndarray, output_path: str, is_mask: bool = False):
        """
        Helper method to process and save a single slice.
        """
        height_factor, width_factor = self.calculate_resize_factors(slice_data)
        resized_slice = ndimage.zoom(slice_data, (width_factor, height_factor, 1), order=1)
        
        if is_mask:
            resized_slice = (resized_slice * 255).astype(np.uint8)  # Ensure binary mask is scaled correctly
        cv2.imwrite(output_path, resized_slice)
        

    def preprocess_train_dataset(self, n: Dict):
        """
        Preprocess one training subject.
        """
        X_path = n["og_input_path"]
        y_path = n["og_mask_path"]
        subject = n['subject']
        SLICE_STRIDE = self.config.slice_stride

        in_nib = nb.load(X_path)
        mask_nib = nb.load(y_path)

        X_img = nb.as_closest_canonical(in_nib).get_fdata()
        y_img = nb.as_closest_canonical(mask_nib).get_fdata()

        img_no = X_img.shape[2] // SLICE_STRIDE

        for i in range(0, img_no, SLICE_STRIDE):
            input_slice = X_img[:, :, i:i+3]  # Stack 3 slices
            mask_slice = np.expand_dims(y_img[:, :, i+2], axis=-1)  # Use the middle slice for mask

            input_path = os.path.join(self.config.preprocess_train_in, f"{subject}_{i}.png")
            mask_path = os.path.join(self.config.preprocess_train_op, f"{subject}_{i}.png")

            self._process_slice(input_slice, input_path)
            self._process_slice(mask_slice, mask_path, is_mask=True)

    def preprocess_test_dataset(self, n: Dict):
        """
        Preprocess one test subject.
        """
        X_path = n["og_input_path"]
        subject = n['subject']
        SLICE_STRIDE = self.config.slice_stride

        X_img = nb.as_closest_canonical(nb.load(X_path)).get_fdata()
        img_no = X_img.shape[2] // SLICE_STRIDE

        for i in range(0, img_no, SLICE_STRIDE):
            input_slice = X_img[:, :, i:i+3]
            input_path = os.path.join(self.config.preprocess_test_in, f"{subject}_{i}.png")
            self._process_slice(input_slice, input_path)

    def preprocess_dataset(self):
        """
        Process the entire dataset.
        """
        try:
            train_layout = BIDSLayout(root=self.config.training_data, validate=False)
            test_layout = BIDSLayout(root=self.config.testing_data, validate=False)

            train_df = self.train_df_process(train_layout.to_df())
            test_df = self.test_df_process(test_layout.to_df())

            logger.info("Training and Testing DataFrames have been created and pre-processed.")

            train_df.apply(lambda x: self.preprocess_train_dataset(x), axis=1)
            test_df.apply(lambda x: self.preprocess_test_dataset(x), axis=1)

            logger.info(f"Pre-processed training data saved at: {self.config.preprocess_train_in}")
            logger.info(f"Pre-processed testing data saved at: {self.config.preprocess_test_in}")

        except Exception as e:
            raise CustomeException(e)