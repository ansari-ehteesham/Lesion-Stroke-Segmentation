import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib as mpl
from bids import BIDSLayout
import matplotlib.pyplot as plt
import sys, random, imageio_ffmpeg
import matplotlib.animation as animation

from lesionSeg.logging import logger 
from lesionSeg.Exception.exception import CustomeException
from lesionSeg.entity import DataValidationEntity


ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path


class DataValidation:
    def __init__(self, config: DataValidationEntity):
        self.config = config

    def animate_slices(self, input_data, mask_data, interval=200):
        n_slices = input_data.shape[2]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        slice_index = 0
        im1 = ax1.imshow(input_data[:, :, slice_index], cmap='gray', origin='lower')
        ax1.set_title(f"T1 Image (Slice {slice_index})")
        ax1.axis('off')
        
        im2 = ax2.imshow(input_data[:, :, slice_index], cmap='gray', origin='lower')
        masked_mask = np.ma.masked_equal(mask_data[:, :, slice_index], 0)
        mask_im = ax2.imshow(masked_mask, cmap='autumn', alpha=0.5, origin='lower')
        ax2.set_title(f"T1 with Lesion Mask (Slice {slice_index})")
        ax2.axis('off')
        
        def update(frame):
            im1.set_array(input_data[:, :, frame])
            ax1.set_title(f"T1 Image (Slice {frame})")
            
            im2.set_array(input_data[:, :, frame])
            masked_mask = np.ma.masked_equal(mask_data[:, :, frame], 0)
            mask_im.set_array(masked_mask)
            ax2.set_title(f"T1 with Lesion Mask (Slice {frame})")
            
            return im1, im2, mask_im
        
        ani = animation.FuncAnimation(
            fig, update, frames=range(n_slices), interval=interval, blit=False, repeat=True
        )
        
        plt.tight_layout()
        return ani
    
    
    def dataset_report(self, train, test):
        tr_len = len(train.get_subjects())
        ts_len = len(test.get_subjects())
        total = tr_len + ts_len

        df = pd.Series({
            "Training Data": tr_len,
            "Testing Data": ts_len,
            "Total Data": total,
        })

        file_dir = self.config.report_file
        df.to_csv(file_dir)
        logger.info(f"Dataset Report: {file_dir}")

        

    def load_nii_file(self, file_path):
        img = nib.load(file_path)
        data = img.get_fdata()
        return data

    def load_dataset(self):
        root_dir = self.config.root_dir
        try:
            training_layout = BIDSLayout(root=f"{root_dir}/Training", validate=False)
            testing_layout = BIDSLayout(root=f"{root_dir}/Testing", validate=False)

            logger.info(f"Total Training Data: {len(training_layout.get_subjects())}")
            logger.info(f"Total Testing Data: {len(testing_layout.get_subjects())}")

            logger.info(f"Extracting a Sample Input and Output Data")

            rand_number = random.randint(a=0, b=654)

            input_sample = training_layout.get(suffix='T1w', extension=['.nii', '.nii.gz'])[rand_number]
            output_sample = training_layout.get(suffix='mask', extension=['.nii', '.nii.gz'])[rand_number]

            logger.info(f"Loading Sample Input: {input_sample.filename}")
            logger.info(f"Loading Sample Output: {output_sample.filename}")

            input_data = self.load_nii_file(input_sample)
            mask_data = self.load_nii_file(output_sample)

            ani = self.animate_slices(input_data, mask_data, interval=200)

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, metadata=dict(artist='abc'), bitrate=1800)

            sample_dir = self.config.animation_file

            ani.save(sample_dir, writer=writer)
            logger.info(f"Saved the sample data at: {sample_dir}")

            self.dataset_report(training_layout, testing_layout)
        except Exception as e:
            raise CustomeException(e)
