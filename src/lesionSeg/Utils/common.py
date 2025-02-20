import cv2
import numpy as np
import tensorflow as tf
import albumentations as A
import sys,os,yaml
from pathlib import Path
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from lesionSeg.logging import logger
from lesionSeg.Exception.exception import CustomeException, catch_ensure_errors



# Read YAML Files
@catch_ensure_errors
@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    """
    It Reads the YAML File

    Arguments:
        file_path: YAML file path 
    
    Return:
        Data: YAML File Data as ConfigBox
    """
    try:
        with open(file=file_path) as yaml_file:
            data = yaml.safe_load(yaml_file)
            logger.info(f"Read YAML File: {file_path}")
            return ConfigBox(data)
    except BoxValueError:
        raise ValueError("YAML FIle is EMPTY")
    except Exception as e:
        raise CustomeException(e)
    

# Create Directories
@catch_ensure_errors
@ensure_annotations
def create_directory(lst_dir:list):
    """
    It create directory
    Arguments:
        lst_dir: list of directory

    Return:
        None
    """
    for dir in lst_dir:
        dir = Path(dir)
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Directory has been Created: {dir}")



class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        input_lst,
        output_lst,
        batch_size=32,
        shuffle=True,
        augmentation=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_lst = input_lst
        self.mask_lst = output_lst
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.n = len(input_lst)

        # Define augmentation pipeline (mask-safe)
        if self.augmentation:
            self.transform = A.Compose([
                # Spatial transforms (applied to BOTH image and mask)
                A.Rotate(limit=30, p=0.5),  # Rotate up to 30 degrees
                A.HorizontalFlip(p=0.5),    # Horizontal flip
                A.VerticalFlip(p=0.5),     # Vertical flip
                A.ShiftScaleRotate(
                    shift_limit=0.1,        # Shift by up to 10% of image size
                    scale_limit=0.2,        # Zoom in/out by up to 20%
                    rotate_limit=0,         # No additional rotation
                    p=0.5
                ),
                A.ElasticTransform(
                    alpha=1,               # Intensity of deformation
                    sigma=50,              # Smoothness of deformation
                    alpha_affine=50,       # Smoothness of affine transformation
                    p=0.5
                ),

                # Pixel-level transforms (applied ONLY to the image)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Blur the image
                A.GaussNoise(var_limit=(10, 50), p=0.5),   # Add Gaussian noise
                A.RGBShift(
                    r_shift_limit=20,       # Shift red channel
                    g_shift_limit=20,       # Shift green channel
                    b_shift_limit=20,       # Shift blue channel
                    p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,           # Maximum number of holes
                    max_height=32,         # Maximum height of a hole
                    max_width=32,          # Maximum width of a hole
                    p=0.5
                ),
            ])
        else:
            self.transform = None

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __get_data(self, files, label='input'):
        imgs = []
        for img_path in files:
            try:
                if label == 'input':
                    # Load and normalize input image
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                else:
                    # Load and binarize mask
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = (img > 0).astype(np.uint8)  # Ensure 0/1 integers

                if img is None:
                    raise ValueError(f"Failed to load {img_path}")
                imgs.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                dummy_shape = (128, 128, 3) if label == 'input' else (128, 128)
                imgs.append(np.zeros(dummy_shape, dtype=np.float32 if label == 'input' else np.uint8))
        return np.array(imgs)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        input_files = [self.input_lst[i] for i in indices]
        mask_files = [self.mask_lst[i] for i in indices]

        X = self.__get_data(input_files, label='input')  # (batch, H, W, 3)
        y = self.__get_data(mask_files, label='mask')    # (batch, H, W)

        if self.transform is not None:
            augmented_X, augmented_y = [], []
            for img, mask in zip(X, y):
                # Apply augmentation (brightness ONLY affects image)
                transformed = self.transform(image=img, mask=mask)
                augmented_X.append(transformed['image'])
                augmented_y.append(transformed['mask'])
            X = np.array(augmented_X)
            y = np.array(augmented_y)

        # Add channel dimension to masks
        y = np.expand_dims(y, axis=-1)
        return X, y

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))