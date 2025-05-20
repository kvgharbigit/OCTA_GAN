import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
import os
import json
import time

from base import HSI_OCTA_Dataset
from circle_crop_utils import crop_and_resize

class HSI_OCTA_Dataset_Cropped(HSI_OCTA_Dataset):
    """Extension of HSI_OCTA_Dataset with circle detection and cropping."""
    
    # Dictionary to keep track of missing files
    missing_files = {}
    
    @classmethod
    def save_missing_files_log(cls, output_dir):
        """
        Save the missing files log to the specified output directory.
        
        Args:
            output_dir: Path to the output directory
        """
        if not cls.missing_files:
            return
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Create a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save to JSON file
        log_path = os.path.join(output_dir, f"missing_files_{timestamp}.json")
        with open(log_path, 'w') as f:
            json.dump(cls.missing_files, f, indent=4)
            
        # Also save a text summary
        txt_path = os.path.join(output_dir, f"missing_files_{timestamp}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Missing Files Log - {timestamp}\n")
            f.write("=======================================\n\n")
            
            if 'hsi' in cls.missing_files:
                f.write(f"HSI Files Missing: {len(cls.missing_files['hsi'])}\n")
                for file in cls.missing_files['hsi']:
                    f.write(f"  - {file}\n")
                f.write("\n")
                
            if 'octa' in cls.missing_files:
                f.write(f"OCTA Files Missing: {len(cls.missing_files['octa'])}\n")
                for file in cls.missing_files['octa']:
                    f.write(f"  - {file}\n")
                f.write("\n")
                
            if 'errors' in cls.missing_files:
                f.write(f"Other Errors: {len(cls.missing_files['errors'])}\n")
                for error in cls.missing_files['errors']:
                    f.write(f"  - {error}\n")
                    
        print(f"Missing files log saved to {log_path} and {txt_path}")

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

    # Dictionary to keep track of missing files
    missing_files = {}
    
    def __getitem__(self, idx: int) -> tuple:
        """Load, preprocess, and return a pair of HSI and OCTA images with circle cropping."""
        # Get file paths for the requested index
        original_idx = self.indices[idx]
        pair = self.file_pairs[original_idx]
        
        # Check if files exist
        hsi_path = pair['hsi']
        octa_path = pair['octa']
        
        # Try to load both modalities with error handling
        try:
            # Check if files exist
            if not os.path.exists(hsi_path):
                # Record missing file if not already recorded
                if 'hsi' not in self.missing_files:
                    self.missing_files['hsi'] = []
                if str(hsi_path) not in self.missing_files['hsi']:
                    self.missing_files['hsi'].append(str(hsi_path))
                    print(f"WARNING: Missing HSI file: {hsi_path}")
                
                # Try the next index
                if idx < len(self) - 1:
                    return self.__getitem__(idx + 1)
                else:
                    return self.__getitem__(0)  # Wrap around to the first item
            
            if not os.path.exists(octa_path):
                # Record missing file if not already recorded
                if 'octa' not in self.missing_files:
                    self.missing_files['octa'] = []
                if str(octa_path) not in self.missing_files['octa']:
                    self.missing_files['octa'].append(str(octa_path))
                    print(f"WARNING: Missing OCTA file: {octa_path}")
                
                # Try the next index
                if idx < len(self) - 1:
                    return self.__getitem__(idx + 1)
                else:
                    return self.__getitem__(0)  # Wrap around to the first item
            
            # Load images
            hsi_img = self._load_hsi(hsi_path)
            octa_img = self._load_octa(octa_path)
            
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
            
        except Exception as e:
            print(f"Error loading files for {pair['patient_id']}: {str(e)}")
            # Record the error
            if 'errors' not in self.missing_files:
                self.missing_files['errors'] = []
            self.missing_files['errors'].append(f"{pair['patient_id']}: {str(e)}")
            
            # Try the next index
            if idx < len(self) - 1:
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)  # Wrap around to the first item