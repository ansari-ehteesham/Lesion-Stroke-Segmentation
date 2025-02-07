import os
import cv2
import sys
import numpy as np
import nibabel as nb
import pandas as pd
from bids import BIDSLayout
from lesionSeg.logging import logger 
from lesionSeg.Exception.exception import CustomeException
from lesionSeg.entity import DataProcessingEntity

import logging
logging.getLogger("nibabel").setLevel(logging.ERROR)


class DataPreprocessing:
    def __init__(self, config: DataProcessingEntity):
        self.config = config

    def normalize_img(self, X):
        """
        Normalize image intensities to the range specified in the config.
        """
        alpha = self.config.img_norms_range[0]
        beta = self.config.img_norms_range[1]
        norm_img = cv2.normalize(X, dst=None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
        return norm_img
    
    def img_resize(self, X, label='input'):
        """
        Resize an image to the target size specified in the config.
        Uses Lanczos interpolation for input images and nearest neighbor for masks.
        """
        img_size = (self.config.img_width, self.config.img_height)
        if label == 'input':
            resized_img = cv2.resize(X, dsize=img_size, interpolation=cv2.INTER_LANCZOS4)
        elif label == 'mask':
            resized_img = cv2.resize(X, dsize=img_size, interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(f"Label -> {label} is Incorrect. Use 'input' or 'mask'.")
        return resized_img
    
    def train_df_process(self, df):
        """
        Process the training dataframe by selecting the necessary columns and merging input and mask files.
        """
        selected_cols = ["path", "subject"]
        df_input = df[df['suffix'] == 'T1w'][selected_cols].rename({'path': 'og_input_path'}, axis=1)
        df_mask = df[df['suffix'] == 'mask'][selected_cols].rename({'path': 'og_mask_path'}, axis=1)
        updt_cols = ['og_input_path', 'og_mask_path', 'subject']
        df_merge = df_input.merge(df_mask, on='subject', how='inner').reset_index(drop=True).reindex(columns=updt_cols)
        return df_merge
    
    def test_df_process(self, df):
        """
        Process the testing dataframe by selecting the necessary columns.
        """
        selected_cols = ["path", "subject"]
        df_input = df[df['suffix'] == 'T1w'][selected_cols].rename({'path': 'og_input_path'}, axis=1)
        return df_input

    def preprocess_train_dataset(self, n):
        """
        Preprocess one training subject: load input and mask volumes, process each slice by resizing 
        and normalizing, then save the processed volumes as NIfTI images.
        """
        X_path = n["og_input_path"]
        y_path = n["og_mask_path"]
        subject = n['subject']
        
        input_dir = self.config.preprocess_train_in
        output_dir = self.config.preprocess_train_op
        processed_input_path = os.path.join(input_dir, f"{subject}_input.nii.gz")
        processed_mask_path = os.path.join(output_dir, f"{subject}_mask.nii.gz")

        in_nib = nb.load(X_path)
        mask_nib = nb.load(y_path)

        ras_in = nb.as_closest_canonical(in_nib)
        ras_mask = nb.as_closest_canonical(mask_nib)

        # Get the data arrays and affine matrices.
        X_img = ras_in.get_fdata()
        y_img = ras_mask.get_fdata()
        affine_input = ras_in.affine
        affine_mask = ras_mask.affine 

        img_no = X_img.shape[2]
        target_height = self.config.img_height
        target_width  = self.config.img_width
        
        # Pre-allocate arrays for processed volumes.
        processed_in_volume = np.empty((target_height, target_width, img_no), dtype=np.float32)
        processed_mask_volume = np.empty((target_height, target_width, img_no), dtype=np.float32)

        for i in range(img_no):
            # Process input slice: resize (with label 'input') then normalize.
            input_img_arr = X_img[:, :, i]
            resized_input = self.img_resize(input_img_arr, label='input')
            norm_input = self.normalize_img(resized_input)
            processed_in_volume[:, :, i] = norm_input

            # Process mask slice: resize using nearest neighbor (with label 'mask').
            mask_img_arr = y_img[:, :, i]
            resized_mask = self.img_resize(mask_img_arr, label='mask')
            processed_mask_volume[:, :, i] = resized_mask

        # Create new NIfTI images using the processed data and the original affines.
        processed_in_nib = nb.Nifti1Image(processed_in_volume, affine_input)
        processed_mask_nib = nb.Nifti1Image(processed_mask_volume, affine_mask)

        nb.save(processed_in_nib, processed_input_path)
        nb.save(processed_mask_nib, processed_mask_path)

        return processed_input_path, processed_mask_path
    
    def preprocess_test_dataset(self, n):
        """
        Preprocess one test subject: load input volume, process each slice by resizing and normalizing,
        then save the processed volume as a NIfTI image.
        """
        X_path = n["og_input_path"]
        subject = n['subject']

        input_dir = self.config.preprocess_test_in
        processed_input_path = os.path.join(input_dir, f"{subject}_input.nii.gz")

        in_nib = nb.load(X_path)
        ras_nib = nb.as_closest_canonical(in_nib)
        X_img = ras_nib.get_fdata()
        affine_input = in_nib.affine

        img_no = X_img.shape[2]
        target_height = self.config.img_height
        target_width  = self.config.img_width
        processed_in_volume = np.empty((target_height, target_width, img_no), dtype=np.float32)

        for i in range(img_no):
            input_img_arr = X_img[:, :, i]
            resized_input = self.img_resize(input_img_arr, label='input')
            norm_input = self.normalize_img(resized_input)
            processed_in_volume[:, :, i] = norm_input
        
        processed_in_nib = nb.Nifti1Image(processed_in_volume, affine_input)
        nb.save(processed_in_nib, processed_input_path)
        return processed_input_path

    def preprocess_dataset(self):
        """
        Process the entire dataset by creating BIDS dataframes, processing training and test files,
        and logging the locations of the processed data.
        """
        try:
            train_root = self.config.training_data
            test_root = self.config.testing_data

            train_layout = BIDSLayout(root=train_root, validate=False)
            test_layout = BIDSLayout(root=test_root, validate=False)

            train_df = train_layout.to_df()
            logger.info("Training Dataframe has been Created")
            test_df = test_layout.to_df()
            logger.info("Testing Dataframe has been Created")

            processed_train_df = self.train_df_process(train_df)
            logger.info("Training Dataframe has been Pre-processed")
            processed_test_df = self.test_df_process(test_df)
            logger.info("Testing Dataframe has been Pre-processed")

            processed_train_df[["processed_input_path", "processed_output_path"]] = processed_train_df.apply(
                lambda x: pd.Series(self.preprocess_train_dataset(x)), axis=1
            )
            logger.info(f"Pre-Processed Input Training Data at: {self.config.preprocess_train_in}")
            logger.info(f"Pre-Processed Mask Training Data at: {self.config.preprocess_train_op}")

            processed_test_df["processed_input_path"] = processed_test_df.apply(
                lambda x: self.preprocess_test_dataset(x), axis=1
            )
            logger.info(f"Pre-Processed Input Testing Data at: {self.config.preprocess_test_in}")
            
            train_csv_dir = self.config.train_csv
            test_csv_dir = self.config.test_csv

            processed_train_df.to_csv(train_csv_dir, index=False)
            logger.info(f'Training Dataset CSV File at: {train_csv_dir}')
            
            processed_test_df.to_csv(test_csv_dir, index=False)
            logger.info(f'Testing Dataset CSV File at: {test_csv_dir}')

            logger.info(f"Total Training Data Instance: {processed_train_df.shape[0]}")
            logger.info(f"Total Testing Data Instance: {processed_test_df.shape[0]}")

        except Exception as e:
            raise CustomeException(e, sys)
