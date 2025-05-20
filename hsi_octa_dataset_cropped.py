import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path

from base import HSI_OCTA_Dataset
from circle_crop_utils import crop_and_resize

class HSI_OCTA_Dataset_Cropped(HSI_OCTA_Dataset):
    """Extension of HSI_OCTA_Dataset with circle detection and cropping."""

    def __init__(self,
                 data_dir: str,
                 transform=None,
                 augment: bool = True,
                 split: str = 'train',
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.4,
                 random_seed: int = 42,
                 target_size: int = 500,
                 crop_padding: int = 10,
                 circle_crop: bool = True,
                 approved_csv_path: str = None,
                 aug_config: dict = None):

        # Initialize the parent class with the approved_csv_path and augmentation config
        super().__init__(data_dir, transform, augment, split,
                         val_ratio, test_ratio, random_seed, target_size,
                         approved_csv_path=approved_csv_path,
                         aug_config=aug_config)

        self.crop_padding = crop_padding
        self.circle_crop = circle_crop
        print(f"Circle cropping is {'enabled' if circle_crop else 'disabled'}")

    def __getitem__(self, idx: int) -> tuple:
        """Load, preprocess, and return a pair of HSI and OCTA images with circle cropping."""
        # Get file paths for the requested index
        pair = self.file_pairs[self.indices[idx]]

        # Load both modalities
        hsi_img = self._load_hsi(pair['hsi'])
        octa_img = self._load_octa(pair['octa'])

        # Ensure HSI is 3D (channels, height, width)
        if hsi_img.dim() == 4:
            hsi_img = hsi_img.squeeze(0)

        # Ensure OCTA is 3D (channels, height, width)
        if octa_img.dim() == 4:
            octa_img = octa_img.squeeze(0)

        # Apply circle cropping and resizing if enabled
        if self.circle_crop:
            hsi_img, octa_img = crop_and_resize(
                hsi_img, octa_img,
                target_size=self.target_size,
                padding=self.crop_padding
            )

        # Apply same augmentation to both images if enabled
        if self.augment and self.aug_transforms:
            # Combine images for consistent augmentation
            combined = torch.cat([hsi_img, octa_img], dim=0)
            combined = self.aug_transforms(combined)

            # Split back into individual modalities
            hsi_img = combined[:hsi_img.shape[0]]
            octa_img = combined[-1:]

        # Apply normalization transforms if provided
        if self.transform:
            hsi_img = self.transform(hsi_img)
            octa_img = self.transform(octa_img)

        return hsi_img, octa_img, pair['patient_id']