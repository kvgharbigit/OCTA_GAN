import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import h5py
from PIL import Image
from pathlib import Path
from torch.nn import init
from typing import List, Tuple, Optional, Dict, TextIO
import sys



# =====================
# Dataset Implementation
# =====================
class HSI_OCTA_Dataset(Dataset):
    """
    Dataset class for paired Hyperspectral Imaging (HSI) and Optical Coherence Tomography Angiography (OCTA) data.

    This class loads HSI and OCTA file paths directly from a CSV file, where each row contains
    the patient ID, eye, HSI file path, and OCTA file path.
    """

    def __init__(self,
                 data_dir: str,
                 transform=None,
                 augment: bool = True,
                 split: str = 'train',
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.4,
                 random_seed: int = 42,
                 target_size: int = 500,
                 approved_csv_path: str = None):
        """
        Initialize the HSI-OCTA dataset using a CSV file for file paths.

        Args:
            data_dir: Directory containing data (not used when approved_csv_path is provided)
            transform: Optional transforms to apply to loaded images
            augment: Whether to apply data augmentation
            split: Dataset split ('train', 'val', or 'test')
            val_ratio: Fraction of data to use for validation
            test_ratio: Fraction of data to use for testing
            random_seed: Random seed for reproducible data splitting
            target_size: Target image size for resizing
            approved_csv_path: Path to CSV file with id_full, eye, hs_file, octa_file columns
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.split = split
        self.target_size = target_size
        self.approved_csv_path = approved_csv_path

        # Check if we're using CSV-based loading or directory-based loading
        if approved_csv_path:
            # CSV-based loading
            self._load_from_csv()
        else:
            # Original directory-based loading
            self._load_from_directory()

        # Split dataset into train, validation, and test sets
        np.random.seed(random_seed)
        indices = np.arange(len(self.file_pairs))
        np.random.shuffle(indices)

        test_size = int(len(indices) * test_ratio)
        val_size = int(len(indices) * val_ratio)
        train_size = len(indices) - test_size - val_size

        # Assign indices based on requested split
        if split == 'train':
            self.indices = indices[:train_size]
        elif split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]

        print(f"Split '{split}' contains {len(self.indices)} samples")

        # Define augmentation pipeline if enabled
        self.aug_transforms = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ) if augment else None

    def _load_from_csv(self):
        """Load file pairs from CSV."""
        try:
            df = pd.read_csv(self.approved_csv_path)
            required_columns = ['id_full', 'eye', 'hs_file', 'octa_file']

            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV is missing required columns: {missing_columns}")

            # Create pairs list
            self.file_pairs = []
            for _, row in df.iterrows():
                # Create a unique patient identifier combining id_full and eye
                patient_id = f"{row['id_full']}_{row['eye']}"

                # Add to pairs list
                self.file_pairs.append({
                    'hsi': Path(row['hs_file']),
                    'octa': Path(row['octa_file']),
                    'patient_id': patient_id
                })

            # For backwards compatibility with original code:
            self.approved_ids = set(df['id_full'].astype(str).tolist())
            self.patient_dirs = []

            print(f"Loaded {len(self.file_pairs)} HSI-OCTA file pairs from CSV")

        except Exception as e:
            print(f"Error loading CSV file from {self.approved_csv_path}: {str(e)}")
            self.file_pairs = []
            raise

    def _load_from_directory(self):
        """Load file pairs using the original directory-based approach."""
        # Load approved IDs if a CSV path is provided but doesn't have the required columns
        self.approved_ids = None
        if self.approved_csv_path:
            self._load_approved_ids()

        # Find all patient directories recursively
        self.patient_dirs = self._find_patient_dirs(self.data_dir)
        print(f"Found {len(self.patient_dirs)} patient directories")

        # Create paired file mapping
        self.file_pairs = self._create_file_pairs(self.patient_dirs)
        print(f"Created {len(self.file_pairs)} HSI-OCTA file pairs")

    def _load_approved_ids(self):
        """Load approved participant IDs from CSV."""
        try:
            approved_df = pd.read_csv(self.approved_csv_path)
            self.approved_ids = set(approved_df['id_full'].astype(str).tolist())
            print(f"Loaded {len(self.approved_ids)} approved participant IDs from {self.approved_csv_path}")
        except Exception as e:
            print(f"Warning: Could not load approved IDs from {self.approved_csv_path}: {str(e)}")
            self.approved_ids = None

    def _find_patient_dirs(self, parent_dir: Path) -> list:
        """
        Recursively find all directories that contain patient data,
        filtering for only approved patient IDs if a list is provided.

        A directory is considered a patient directory if it contains
        both HSI (.h5) and OCTA (.tiff) files.

        Args:
            parent_dir: Parent directory to search

        Returns:
            List of patient directory paths
        """
        patient_dirs = []

        # Process all subdirectories recursively
        for dir_path in parent_dir.glob('**/'):
            # Check if this directory contains HSI and OCTA files
            hs_files = list(dir_path.glob('*.h5'))
            octa_files = list(dir_path.glob('*RetinaAngiographyEnface*.tiff'))

            if hs_files and octa_files:
                # Extract patient ID from directory name
                patient_id = dir_path.name

                # Only include if patient is approved, or if no approval list exists
                if self.approved_ids is None or patient_id in self.approved_ids:
                    patient_dirs.append(dir_path)

        # Print statistics about approved vs. found patients if we have an approval list
        if self.approved_ids is not None:
            found_ids = set(dir_path.name for dir_path in patient_dirs)
            missing_ids = self.approved_ids - found_ids

            print(f"Found {len(patient_dirs)} approved patient directories")
            print(f"Missing {len(missing_ids)} approved patients")

            if missing_ids and len(missing_ids) <= 10:
                print(f"Missing patients: {missing_ids}")
            elif missing_ids:
                print(f"First 10 missing patients: {list(missing_ids)[:10]}...")

        return patient_dirs

    def _create_file_pairs(self, patient_dirs: list) -> list:
        """
        Create pairs of corresponding HSI and OCTA files.

        This method searches each patient directory for HSI files (both C1 and D1 patterns)
        and OCTA files, then pairs them. It prefers C1 files over D1 when both are present.

        Args:
            patient_dirs: List of patient directory paths to search

        Returns:
            List of dictionaries containing paired HSI and OCTA file paths
        """
        pairs = []
        for patient_dir in patient_dirs:
            # Find all HSI files with either C1 or D1 pattern
            c1_files = list(patient_dir.glob('*C1*.h5'))
            d1_files = list(patient_dir.glob('*D1*.h5'))
            octa_files = list(patient_dir.glob('*RetinaAngiographyEnface*.tiff'))

            # Determine which HSI file to use (prefer C1 if available)
            hs_file = None
            if len(c1_files) == 1:
                hs_file = c1_files[0]
            elif len(d1_files) == 1:
                hs_file = d1_files[0]
            elif len(c1_files) > 1:
                print(f"Warning: Multiple C1 files in {patient_dir}, using first one")
                hs_file = c1_files[0]
            elif len(d1_files) > 1:
                print(f"Warning: Multiple D1 files in {patient_dir}, using first one")
                hs_file = d1_files[0]

            # Create pair if both HSI and OCTA files are available
            if hs_file and len(octa_files) == 1:
                # Extract the patient ID from the directory path
                # Use the last part of the path as the patient ID if it doesn't have a specific pattern
                patient_id = patient_dir.name

                pairs.append({
                    'hsi': hs_file,
                    'octa': octa_files[0],
                    'patient_id': patient_id
                })
            else:
                if not hs_file:
                    print(f"Warning: No suitable HSI files (C1 or D1) found in {patient_dir}")
                if len(octa_files) != 1:
                    print(f"Warning: Missing or multiple OCTA files in {patient_dir}")

        return pairs

    def _load_hsi(self, hsi_path: Path) -> torch.Tensor:
        """
        Load and preprocess HSI data, using every third wavelength.

        This function loads HSI data from an h5 file, selects every third wavelength band
        to reduce dimensionality, normalizes the data to [0,1] range, and resizes to the
        target spatial dimensions.

        Args:
            hsi_path: Path to the HSI h5 file

        Returns:
            Preprocessed HSI tensor with shape [31, H, W]
        """
        with h5py.File(hsi_path, 'r') as hs_file:
            hsi_img = hs_file['Cube/Images'][:]

            # Get original number of wavelengths
            original_wavelengths = hsi_img.shape[0]

            # Take every third wavelength to reduce dimensionality
            hsi_img = hsi_img[::3]

            # Check if we have exactly 31 wavelengths after taking every 3rd one
            actual_wavelengths = hsi_img.shape[0]
            expected_wavelengths = 31

            if actual_wavelengths != expected_wavelengths:
                # Handle the case where we don't have exactly 31 wavelengths
                print(f"Warning: Expected {expected_wavelengths} wavelengths but got {actual_wavelengths} "
                      f"after taking every 3rd from {original_wavelengths} wavelengths in {hsi_path.name}")

                if actual_wavelengths > expected_wavelengths:
                    # If we have too many, trim to exactly 31
                    hsi_img = hsi_img[:expected_wavelengths]
                    print(f"  - Trimmed to first {expected_wavelengths} wavelengths")
                else:
                    # If we have too few, pad with zeros to reach 31
                    padding_needed = expected_wavelengths - actual_wavelengths
                    # Create padding array with same spatial dimensions and proper data type
                    padding = np.zeros((padding_needed, *hsi_img.shape[1:]), dtype=hsi_img.dtype)
                    # Concatenate with original data
                    hsi_img = np.concatenate([hsi_img, padding], axis=0)
                    print(f"  - Padded with {padding_needed} zero wavelength bands")

            hsi_img = torch.tensor(hsi_img, dtype=torch.float32)  # Shape: (31, H, W)

            # Normalize before resizing
            if hsi_img.max() > 1.0:
                hsi_img = hsi_img / hsi_img.max()

            # Resize each spectral band to target size
            if hsi_img.shape[1] != self.target_size or hsi_img.shape[2] != self.target_size:
                resized_hsi = torch.zeros((hsi_img.shape[0], self.target_size, self.target_size))
                for i in range(hsi_img.shape[0]):
                    resized_hsi[i] = F.interpolate(
                        hsi_img[i].unsqueeze(0).unsqueeze(0),
                        size=(self.target_size, self.target_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                hsi_img = resized_hsi

            # Final check to ensure correct dimensions
            assert hsi_img.shape[
                       0] == expected_wavelengths, f"HSI tensor should have {expected_wavelengths} channels, but has {hsi_img.shape[0]}"

            return hsi_img

    def _load_octa(self, octa_path: Path) -> torch.Tensor:
        """
        Load and preprocess OCTA image with resizing to target size.

        This function loads an OCTA image from a tiff file, converts to grayscale,
        resizes to the target dimensions, and normalizes to [0,1] range.

        Args:
            octa_path: Path to the OCTA tiff file

        Returns:
            Preprocessed OCTA tensor with shape [1, H, W]
        """
        octa_img = Image.open(octa_path).convert('L')  # Convert to grayscale

        # Resize to target size
        if octa_img.size != (self.target_size, self.target_size):
            octa_img = octa_img.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)

        # Convert to numpy array and normalize to [0,1] range
        octa_img = np.array(octa_img, dtype=np.float32) / 255.0

        # Convert to torch tensor and add channel dimension
        octa_img = torch.tensor(octa_img, dtype=torch.float32).unsqueeze(0)
        return octa_img

    def __len__(self) -> int:
        """Return the number of samples in the dataset split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        """
        Load and preprocess a pair of HSI and OCTA images.

        Args:
            idx: Index of the sample to load

        Returns:
            Tuple containing:
              - HSI tensor with shape [31, H, W]
              - OCTA tensor with shape [1, H, W]
              - Patient ID string
        """
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


# =====================
# Generator Architecture
# =====================
class Generator(nn.Module):
    """
    Generator network for translating HSI to OCTA images.

    This architecture uses 3D convolutions to process the spectral dimension of HSI data,
    employs skip connections to preserve spatial information, and produces a single-channel
    OCTA output.
    """

    def __init__(self, spectral_channels: int = 31, model_size: str = "medium"):
        """
        Initialize the generator network.

        Args:
            spectral_channels: Number of spectral channels in the input HSI data
            model_size: Size of the model ('small', 'medium', or 'large')
        """
        super(Generator, self).__init__()
        
        # Define filter sizes based on model size
        if model_size == "small":
            initial_filters = 16
            mid_filters = 32
            max_filters = 64
            # More balanced encoder-decoder progression
            decoder_mid_filters = 48  # Between initial and max
        elif model_size == "medium":
            initial_filters = 32
            mid_filters = 64
            max_filters = 128
            # More balanced encoder-decoder progression
            decoder_mid_filters = 96  # Between initial and max
        elif model_size == "large":
            initial_filters = 64
            mid_filters = 128
            max_filters = 320  # Increased for better balance with discriminator
            # More balanced encoder-decoder progression
            decoder_mid_filters = 192  # Between initial and max
        else:
            raise ValueError(f"Invalid model size: {model_size}. Choose from 'small', 'medium', or 'large'")
            
        print(f"Initializing Generator with {model_size} size: {initial_filters}/{mid_filters}/{max_filters} filters")

        # Encoder layers - process 3D input data with spectral dimension reduction
        self.encoder = nn.ModuleList([
            # [B, 1, 31, 500, 500] -> [B, initial_filters, 16, 500, 500]
            # First encoder layer reduces spectral dimensions by half
            nn.Conv3d(1, initial_filters, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),

            # [B, initial_filters, 16, 500, 500] -> [B, mid_filters, 8, 500, 500]
            # Second encoder layer further reduces spectral dimensions
            nn.Conv3d(initial_filters, mid_filters, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),

            # [B, mid_filters, 8, 500, 500] -> [B, max_filters, 4, 500, 500]
            # Third encoder layer creates deep features with reduced spectral dimensions
            nn.Conv3d(mid_filters, max_filters, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        ])

        # Skip connections to preserve spatial information
        self.skip_connections = nn.ModuleList([
            # Process features from first encoder layer for skip connection
            nn.Sequential(
                nn.Conv3d(initial_filters, initial_filters, 1),  # 1x1x1 convolution to refine features
                nn.BatchNorm3d(initial_filters),
                nn.ReLU()
            ),
            # Process features from second encoder layer for skip connection
            nn.Sequential(
                nn.Conv3d(mid_filters, mid_filters, 1),  # 1x1x1 convolution to refine features
                nn.BatchNorm3d(mid_filters),
                nn.ReLU()
            )
        ])

        # Calculate the concatenated channel sizes for reduction layers
        # First concatenation: Output of last encoder (max_filters) + mid_filters from skip connection
        concat1_channels = max_filters + mid_filters
        # Second concatenation: Output of first decoder (decoder_mid_filters) + initial_filters from skip connection
        concat2_channels = decoder_mid_filters + initial_filters

        # Channel reduction layers after concatenation with skip connections
        # The first reduction layer output needs to match the first decoder input
        self.reduction_layers = nn.ModuleList([
            # Reduce channels after first skip concatenation
            nn.Sequential(
                nn.Conv2d(concat1_channels, max_filters, 1),  # (max_filters + mid_filters) -> max_filters
                nn.BatchNorm2d(max_filters),
                nn.ReLU()
            ),
            # Reduce channels after second skip concatenation
            nn.Sequential(
                nn.Conv2d(concat2_channels, decoder_mid_filters, 1),  # (decoder_mid_filters + initial_filters) -> decoder_mid_filters
                nn.BatchNorm2d(decoder_mid_filters),
                nn.ReLU()
            )
        ])

        # Decoder layers with more balanced progression
        self.decoder = nn.ModuleList([
            # First decoder block - maintain high-level features
            nn.Sequential(
                nn.Conv2d(max_filters, max_filters, 3, 1, 1),  # max_filters -> max_filters
                nn.BatchNorm2d(max_filters),
                nn.ReLU(),
                nn.Conv2d(max_filters, decoder_mid_filters, 3, 1, 1),  # max_filters -> decoder_mid_filters
                nn.BatchNorm2d(decoder_mid_filters),
                nn.ReLU()
            ),
            # Second decoder block - more gradual progression
            nn.Sequential(
                nn.Conv2d(decoder_mid_filters, decoder_mid_filters, 3, 1, 1),  # decoder_mid_filters -> decoder_mid_filters
                nn.BatchNorm2d(decoder_mid_filters),
                nn.ReLU(),
                nn.Conv2d(decoder_mid_filters, mid_filters, 3, 1, 1),  # decoder_mid_filters -> mid_filters
                nn.BatchNorm2d(mid_filters),
                nn.ReLU()
            ),
            # Third decoder block - more balanced reduction
            nn.Sequential(
                nn.Conv2d(mid_filters, mid_filters, 3, 1, 1),  # mid_filters -> mid_filters
                nn.BatchNorm2d(mid_filters),
                nn.ReLU(),
                nn.Conv2d(mid_filters, initial_filters, 3, 1, 1),  # mid_filters -> initial_filters
                nn.BatchNorm2d(initial_filters),
                nn.ReLU()
            ),
            # Final output layer - generate single-channel OCTA
            nn.Sequential(
                nn.Conv2d(initial_filters, 1, 3, 1, 1),  # initial_filters -> 1
                nn.Tanh()  # Output in range [-1, 1]
            )
        ])

    @staticmethod
    def test_dimensions():
        """
        Test method to verify tensor dimensions through the network for all model sizes.
        This helps catch any dimension mismatches before running the model.
        """
        import torch
        print("\n===== Testing Generator Dimensions =====")
        
        # Test all model sizes
        for size in ["small", "medium", "large"]:
            print(f"\nModel size: {size}")
            
            # Create a dummy input tensor
            batch_size = 2
            spectral_channels = 31
            height = width = 500
            dummy_input = torch.randn(batch_size, spectral_channels, height, width)
            
            # Create model
            model = Generator(spectral_channels=spectral_channels, model_size=size)
            
            # Enable debug mode temporarily (the forward pass will print dimensions)
            model._debug = True
            
            try:
                # Run a forward pass
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"✅ Success! Output shape: {output.shape}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
            
            # Disable debug mode
            model._debug = False
        
        print("\n===== Dimension Test Complete =====")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            x: Input HSI tensor with shape [B, 31, 500, 500]

        Returns:
            Generated OCTA tensor with shape [B, 1, 500, 500]
        """
        # For debugging dimension issues
        debug = getattr(self, '_debug', False)
        
        # Input: [B, 31, 500, 500]
        skips = []
        x = x.unsqueeze(1)  # [B, 1, 31, 500, 500] - Add channel dimension for 3D conv
        if debug: print(f"Input shape: {x.shape}")
        
        # Encoder path with skip connections
        for i, enc_layer in enumerate(self.encoder):
            x = F.relu(enc_layer(x))
            if debug: print(f"Encoder {i} output shape: {x.shape}")
            
            if i < len(self.skip_connections):
                skip = self.skip_connections[i](x)
                skips.append(skip)
                if debug: print(f"Skip connection {i} shape: {skip.shape}")

        # Collapse spectral dimension with max pooling
        x = x.max(dim=2)[0]  # [B, max_filters, 500, 500] - Spectral dimension collapsed
        if debug: print(f"After max pooling shape: {x.shape}")

        # Decoder path with skip connections
        # First decoder layer - handle skip connection from last encoder layer
        skip = skips[-1].max(dim=2)[0]  # Collapse spectral dimension
        if debug: print(f"First skip connection shape after max pooling: {skip.shape}")
        
        # Important step: Concatenate encoder output with skip connection
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        if debug: print(f"After first concatenation shape: {x.shape}")
        
        # Apply first reduction layer to match expected decoder input channels
        x = self.reduction_layers[0](x)  # Reduce channels to max_filters
        if debug: print(f"After first reduction shape: {x.shape}")
        
        # Apply first decoder layer
        x = self.decoder[0](x)  # Output has decoder_mid_filters channels
        if debug: print(f"After first decoder shape: {x.shape}")
        
        # Second decoder layer - handle skip connection from second-to-last encoder layer
        skip = skips[-2].max(dim=2)[0]  # Collapse spectral dimension
        if debug: print(f"Second skip connection shape after max pooling: {skip.shape}")
        
        # Concatenate first decoder output with second skip connection
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        if debug: print(f"After second concatenation shape: {x.shape}")
        
        # Apply second reduction layer
        x = self.reduction_layers[1](x)  # Reduce channels to decoder_mid_filters
        if debug: print(f"After second reduction shape: {x.shape}")
        
        # Apply second decoder layer
        x = self.decoder[1](x)  # Output has mid_filters channels
        if debug: print(f"After second decoder shape: {x.shape}")
        
        # Remaining decoder layers (no more skip connections)
        for i in range(2, len(self.decoder)-1):
            x = self.decoder[i](x)
            if debug: print(f"After decoder {i} shape: {x.shape}")

        # Final output layer
        output = self.decoder[-1](x)
        if debug: print(f"Output shape: {output.shape}")
        
        return output


# =====================
# Discriminator Architecture
# =====================
class Discriminator(nn.Module):
    """
    PatchGAN discriminator for differentiating between real and generated OCTA images.

    This discriminator classifies overlapping image patches rather than the whole image,
    allowing it to focus on local texture details rather than global structure.
    """

    def __init__(self, model_size: str = "medium"):
        """
        Initialize the PatchGAN discriminator network.
        
        Args:
            model_size: Size of the model ('small', 'medium', or 'large')
        """
        super(Discriminator, self).__init__()
        
        # Define filter sizes based on model size with improved balance
        if model_size == "small":
            initial_filters = 32
            mid_filters = 48
            large_filters = 96
            max_filters = 128  # Reduced from 256 for better generator/discriminator balance
        elif model_size == "medium":
            initial_filters = 64
            mid_filters = 96
            large_filters = 192
            max_filters = 256  # Reduced from 512 for better generator/discriminator balance
        elif model_size == "large":
            initial_filters = 96
            mid_filters = 192
            large_filters = 320
            max_filters = 512  # Reduced from 1024 for better generator/discriminator balance
        else:
            raise ValueError(f"Invalid model size: {model_size}. Choose from 'small', 'medium', or 'large'")
            
        print(f"Initializing Discriminator with {model_size} size: {initial_filters}/{mid_filters}/{large_filters}/{max_filters} filters")

        def discriminator_block(in_channels: int, out_channels: int,
                                normalize: bool = True) -> List[nn.Module]:
            """
            Helper function to create a block of layers for the discriminator.

            Args:
                in_channels: Number of input channels
                out_channels: Number of output channels
                normalize: Whether to include batch normalization

            Returns:
                List of layers for the block
            """
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # For 500x500 input, output will be 31x31 patches
        self.model = nn.Sequential(
            *discriminator_block(1, initial_filters, normalize=False),  # No normalization on first layer
            *discriminator_block(initial_filters, mid_filters),
            *discriminator_block(mid_filters, large_filters),
            *discriminator_block(large_filters, max_filters),
            nn.Conv2d(max_filters, 1, 4, padding=1),
            nn.Sigmoid()  # Output in range [0, 1] - probability that patch is real
        )

    @staticmethod
    def test_dimensions():
        """
        Test method to verify tensor dimensions through the discriminator for all model sizes.
        This helps catch any dimension mismatches before running the model.
        """
        import torch
        print("\n===== Testing Discriminator Dimensions =====")
        
        # Test all model sizes
        for size in ["small", "medium", "large"]:
            print(f"\nModel size: {size}")
            
            # Create a dummy input tensor (simulating OCTA images)
            batch_size = 2
            channels = 1
            height = width = 500
            dummy_input = torch.randn(batch_size, channels, height, width)
            
            # Create model
            model = Discriminator(model_size=size)
            
            try:
                # Run a forward pass
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"✅ Success! Input shape: {dummy_input.shape}, Output shape: {output.shape}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print("\n===== Dimension Test Complete =====")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            x: Input OCTA tensor with shape [B, 1, 500, 500]

        Returns:
            Patch-wise classification results with shape [B, 1, 30, 30]
        """
        return self.model(x)


# =====================
# Perceptual Loss
# =====================
class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.

    This loss compares the feature representations of real and generated images
    using a pre-trained VGG16 network, encouraging the generator to produce
    images that are perceptually similar to the targets.
    """

    def __init__(self, layers: List[str] = ['3', '8', '15', '22']):
        """
        Initialize the perceptual loss module.

        Args:
            layers: List of VGG16 layer indices to extract features from
        """
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        vgg.eval()

        # Normalization for ImageNet-pretrained VGG
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # Create feature extractors for each specified layer
        self.feature_extractors = nn.ModuleList()
        previous_layer = 0

        for layer_idx in layers:
            layer_idx = int(layer_idx)
            layers = []
            for module in list(vgg.features.children())[previous_layer:layer_idx]:
                if isinstance(module, nn.ReLU):
                    # Replace in-place ReLU with regular ReLU to avoid modifying inputs
                    layers.append(nn.ReLU(inplace=False))
                else:
                    layers.append(module)
            sequential = nn.Sequential(*layers)
            self.feature_extractors.append(sequential)
            previous_layer = layer_idx

        # Freeze parameters - we don't want to train VGG
        for param in self.parameters():
            param.requires_grad = False

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert single-channel image to 3 channels and normalize for VGG.

        Args:
            x: Single-channel image tensor

        Returns:
            Preprocessed tensor for VGG input
        """
        # Clone input to avoid in-place modifications
        x = x.clone()

        # Repeat the single channel 3 times for RGB input to VGG
        x = x.repeat(1, 3, 1, 1)

        # Normalize with ImageNet statistics
        x = self.normalize(x)
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between generated and target images.

        Args:
            x: Generated image tensor
            y: Target image tensor

        Returns:
            Perceptual loss value
        """
        # Preprocess inputs for VGG
        x = self._preprocess(x)
        y = self._preprocess(y)

        loss = 0
        for extractor in self.feature_extractors:
            # Extract features
            x = extractor(x)
            y = extractor(y)

            # Normalize features safely without in-place operations
            x_norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8
            y_norm = torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8

            x = x / x_norm
            y = y / y_norm

            # Compute MSE loss on normalized features
            loss = loss + F.mse_loss(x, y)

        return loss


# =====================
# SSIM Loss
# =====================
class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for preserving structural information.

    SSIM measures the similarity between two images based on luminance, contrast,
    and structure, providing a more perceptually relevant loss than pixel-wise losses.
    """

    def __init__(self, window_size: int = 11):
        """
        Initialize the SSIM loss module.

        Args:
            window_size: Size of the Gaussian window
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.register_buffer('window', self._create_window())
        self.C1 = 0.01 ** 2  # Constant to stabilize division with weak denominator
        self.C2 = 0.03 ** 2  # Constant to stabilize division with weak denominator

    def _create_window(self) -> torch.Tensor:
        """
        Creates a Gaussian window for SSIM computation.

        Returns:
            2D Gaussian kernel as a tensor
        """
        # Create coordinates
        coords = torch.arange(self.window_size, dtype=torch.float32)
        center = (self.window_size - 1) / 2.0

        # Create 1D Gaussian kernel
        sigma = 1.5
        gauss = torch.exp(-((coords - center) ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # Create 2D Gaussian kernel
        window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        window = window / window.sum()

        return window.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss between generated and target images.

        Args:
            x: Generated image tensor
            y: Target image tensor

        Returns:
            SSIM loss value (1 - SSIM)
        """
        # Clone inputs to avoid in-place modifications
        x = x.clone()
        y = y.clone()

        # Expand window to match input channels
        window = self.window.expand(x.size(1), 1, self.window_size, self.window_size)

        # Compute means using convolution with Gaussian window
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=x.size(1))
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=y.size(1))

        # Compute variances and covariance
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=self.window_size // 2, groups=x.size(1)) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=self.window_size // 2, groups=y.size(1)) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=x.size(1)) - mu_xy

        # Compute SSIM
        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))

        # Return loss (1 - SSIM) since we want to minimize
        return 1 - ssim_map.mean()


# =====================
# Training Configuration
# =====================
class TrainingConfig:
    """
    Configuration class for training hyperparameters and settings.

    This class centralizes all training hyperparameters for easy modification.
    """

    def __init__(self):
        """Initialize with default training parameters."""
        # Training hyperparameters
        self.num_epochs = 200
        self.batch_size = 8
        self.learning_rate = 0.0002  # Standard learning rate for GANs
        self.beta1 = 0.5  # Adam optimizer beta1
        self.beta2 = 0.999  # Adam optimizer beta2

        # Loss weights - balance different loss components
        self.lambda_pixel = 100.0  # L1 loss weight
        self.lambda_perceptual = 10.0  # Perceptual loss weight
        self.lambda_ssim = 5.0  # SSIM loss weight
        self.lambda_adv = 1.0  # Adversarial loss weight

        # Training settings
        self.save_interval = 10  # Save model every N epochs
        self.print_interval = 100  # Print progress every N batches
        self.validate_interval = 5  # Run validation every N epochs

        # Optimizer settings
        self.weight_decay = 1e-4  # L2 regularization
        self.gradient_clip = 1.0  # Gradient clipping threshold

        # Learning rate scheduling
        self.lr_decay_start = 100  # Start learning rate decay after this epoch
        self.lr_decay_factor = 0.1  # Multiply learning rate by this factor
        self.lr_decay_interval = 50  # Apply decay every N epochs


# =====================
# Model Initialization Functions
# =====================
def init_weights(model: nn.Module) -> None:
    """
    Initialize network weights using Xavier initialization.

    This helps with faster convergence at the beginning of training.

    Args:
        model: Neural network module to initialize
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        # Initialize convolutional layers with Xavier/Glorot initialization
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.constant_(model.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        # Initialize batch norm layers with normal distribution and zeros for bias
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0.0)


def get_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration dictionary

    Returns:
        Learning rate scheduler
    """
    scheduler_config = config.get('lr_scheduler', {})
    scheduler_type = scheduler_config.get('type', 'StepLR')

    if scheduler_type == 'StepLR':
        # Standard step scheduler: reduces LR at fixed intervals
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('lr_decay_interval', 50),
            gamma=scheduler_config.get('lr_decay_factor', 0.1),
            last_epoch=-1
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        # Plateau scheduler: reduces LR when a metric stops improving
        # Create parameters dictionary and filter out unsupported parameters
        scheduler_kwargs = {
            'optimizer': optimizer,
            'mode': scheduler_config.get('mode', 'min'),
            'factor': scheduler_config.get('factor', 0.1),
            'patience': scheduler_config.get('patience', 10),
            'min_lr': scheduler_config.get('min_lr', 1e-6),
            'threshold': scheduler_config.get('threshold', 1e-4),
            'cooldown': scheduler_config.get('cooldown', 0),
            'threshold_mode': scheduler_config.get('threshold_mode', 'rel')
        }
        
        # Try to add verbose if supported by this PyTorch version
        try:
            # Create a small test scheduler to check if verbose is supported
            test_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                verbose=True
            )
            # If no error, verbose is supported
            scheduler_kwargs['verbose'] = scheduler_config.get('verbose', True)
        except TypeError:
            # verbose parameter is not supported in this PyTorch version
            print("Note: 'verbose' parameter is not supported in your PyTorch version, ignoring it.")
            # If verbose was set to True in config, print a message about its behavior
            if scheduler_config.get('verbose', True):
                print("Learning rate changes will not be automatically printed. Check metrics for LR changes.")
        
        # Create and return the scheduler with all supported parameters
        return torch.optim.lr_scheduler.ReduceLROnPlateau(**scheduler_kwargs)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


# =====================
# Training Utilities
# =====================

def save_checkpoint(state: Dict, filename: str) -> None:
    """
    Save training checkpoint to disk.

    This allows resuming training from a saved state and keeps track of the best models.

    Args:
        state: Dictionary containing model and training state (weights, optimizer state, epoch, etc.)
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    """
    Load training checkpoint from disk.

    Restores model weights and optionally optimizer state from a saved checkpoint,
    allowing training to resume from a specific point.

    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)

    Returns:
        The epoch number where training left off
    """
    # Load the checkpoint from disk
    checkpoint = torch.load(filename)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the epoch where we left off
    return checkpoint['epoch']


# =====================
# Model Structure Printing Functions
# =====================

def print_model_structure(model: nn.Module, file: TextIO = sys.stdout, indent: int = 0, depth_first: bool = False) -> None:
    """
    Print a detailed hierarchical representation of a PyTorch model's structure.
    
    This function prints the model's architecture, showing nested modules, 
    parameters, shapes, and total parameter counts. The structure can be printed 
    in breadth-first or depth-first order.
    
    Args:
        model: The PyTorch model to visualize
        file: File object to write to (default: sys.stdout)
        indent: Initial indentation level (used for recursive calls)
        depth_first: Whether to use depth-first traversal (default: False)
    """
    # Get total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Safe printing function to handle encoding issues
    def safe_print(text, file=file):
        try:
            print(text, file=file)
        except UnicodeEncodeError:
            # Replace problematic characters with '?'
            print(text.encode('ascii', 'replace').decode('ascii'), file=file)
    
    # Print model class name and parameter summary
    if indent == 0:
        safe_print(f"\n{'=' * 80}")
        safe_print(f"MODEL STRUCTURE: {model.__class__.__name__}")
        safe_print(f"{'=' * 80}")
        safe_print(f"Total parameters: {total_params:,}")
        safe_print(f"Trainable parameters: {trainable_params:,}")
        safe_print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        safe_print(f"{'-' * 80}")
    
    # Get all direct child modules
    children = list(model.named_children())
    
    # If no children, print parameter details
    if not children:
        for name, param in model.named_parameters(recurse=False):
            shape_str = 'x'.join(str(x) for x in param.shape)
            safe_print(f"{' ' * indent}├── Parameter {name}: {shape_str} ({param.numel():,} elements)")
        return
    
    # Process each child module
    for i, (name, child) in enumerate(children):
        # Get child parameter count
        child_params = sum(p.numel() for p in child.parameters())
        
        # Determine prefix based on position
        if i == len(children) - 1:
            prefix = '└──'
            next_indent = indent + 4
        else:
            prefix = '├──'
            next_indent = indent + 4
        
        # Print child module info
        child_modules = sum(1 for _ in child.modules()) - 1  # -1 to not count self
        safe_print(f"{' ' * indent}{prefix} {name} ({child.__class__.__name__}): {child_params:,} params, {child_modules} sub-modules")
        
        # Recursively print nested modules (either immediately for depth-first, or after siblings for breadth-first)
        if depth_first:
            print_model_structure(child, file, next_indent, depth_first)
    
    # Process children breadth-first after all direct children are listed
    if not depth_first:
        for i, (name, child) in enumerate(children):
            next_indent = indent + 4
            print_model_structure(child, file, next_indent, depth_first)


def get_model_summary_string(model: nn.Module, input_shape: tuple = None, depth_first: bool = False) -> str:
    """
    Generate a string representation of the model structure.
    
    Args:
        model: The PyTorch model to summarize
        input_shape: Optional input shape to calculate output shapes (e.g., (1, 31, 500, 500))
        depth_first: Whether to use depth-first traversal (default: False)
        
    Returns:
        String representation of the model structure
    """
    import io
    
    # Create a string buffer to capture the output
    buffer = io.StringIO()
    
    # Print model structure to the buffer
    print_model_structure(model, buffer, depth_first=depth_first)
    
    # Add input/output shape information if provided
    if input_shape is not None:
        try:
            # Create a dummy input tensor
            device = next(model.parameters()).device
            dummy_input = torch.zeros(input_shape).to(device)
            
            # Temporarily disable gradient computation for efficiency
            with torch.no_grad():
                output = model(dummy_input)
            
            # Add input/output information
            buffer.write(f"\n{'-' * 80}\n")
            buffer.write(f"Input shape: {input_shape}\n")
            buffer.write(f"Output shape: {tuple(output.shape)}\n")
            buffer.write(f"{'-' * 80}\n")
        except Exception as e:
            buffer.write(f"\nError calculating output shape: {str(e)}\n")
    
    # Get the buffer content as a string
    result = buffer.getvalue()
    buffer.close()
    
    return result


def save_model_structure(model: nn.Module, filename: str, input_shape: tuple = None, depth_first: bool = False) -> None:
    """
    Save the model structure to a text file.
    
    Args:
        model: The PyTorch model to save
        filename: Path to save the structure information
        input_shape: Optional input shape to calculate output shapes
        depth_first: Whether to use depth-first traversal (default: False)
    """
    try:
        # Try with UTF-8 encoding first
        with open(filename, 'w', encoding='utf-8') as f:
            # Print model structure to the file
            print_model_structure(model, f, depth_first=depth_first)
            
            # Add input/output shape information if provided
            if input_shape is not None:
                try:
                    # Create a dummy input tensor
                    device = next(model.parameters()).device
                    dummy_input = torch.zeros(input_shape).to(device)
                    
                    # Temporarily disable gradient computation for efficiency
                    with torch.no_grad():
                        output = model(dummy_input)
                    
                    # Add input/output information
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"Input shape: {input_shape}\n")
                    f.write(f"Output shape: {tuple(output.shape)}\n")
                    f.write(f"{'-' * 80}\n")
                except Exception as e:
                    f.write(f"\nError calculating output shape: {str(e)}\n")
    except UnicodeEncodeError:
        # If UTF-8 fails, try with ASCII encoding and replace problematic characters
        with open(filename, 'w', encoding='ascii', errors='replace') as f:
            # Print model structure to the file
            print_model_structure(model, f, depth_first=depth_first)
            
            # Add input/output shape information if provided
            if input_shape is not None:
                try:
                    # Create a dummy input tensor
                    device = next(model.parameters()).device
                    dummy_input = torch.zeros(input_shape).to(device)
                    
                    # Temporarily disable gradient computation for efficiency
                    with torch.no_grad():
                        output = model(dummy_input)
                    
                    # Add input/output information
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"Input shape: {input_shape}\n")
                    f.write(f"Output shape: {tuple(output.shape)}\n")
                    f.write(f"{'-' * 80}\n")
                except Exception as e:
                    f.write(f"\nError calculating output shape: {str(e)}\n")
    
    print(f"Model structure saved to {filename}")