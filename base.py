import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
import numpy as np
import h5py
from PIL import Image
from pathlib import Path
from torch.nn import init
from typing import List, Tuple, Optional, Dict


# =====================
# Dataset Implementation
# =====================
class HSI_OCTA_Dataset(Dataset):
    """
    Dataset class for paired Hyperspectral Imaging (HSI) and Optical Coherence Tomography Angiography (OCTA) data.

    This class handles loading, preprocessing, and augmentation of paired HSI and OCTA retinal images.
    HSI data is loaded from .h5 files, with each HSI containing multiple spectral bands.
    OCTA data is loaded from .tiff files and contains angiography information.
    """

    def __init__(self,
                 data_dir: str,
                 transform=None,
                 augment: bool = True,
                 split: str = 'train',
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.4,
                 random_seed: int = 42,
                 target_size: int = 500):
        """
        Initialize the HSI-OCTA dataset.

        Args:
            data_dir: Directory containing subdirectories with patient data
            transform: Optional transforms to apply to loaded images
            augment: Whether to apply data augmentation
            split: Dataset split ('train', 'val', or 'test')
            val_ratio: Fraction of data to use for validation
            test_ratio: Fraction of data to use for testing
            random_seed: Random seed for reproducible data splitting
            target_size: Target image size for resizing
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.split = split
        self.target_size = target_size

        # Find all patient directories recursively
        self.patient_dirs = self._find_patient_dirs(self.data_dir)
        print(f"Found {len(self.patient_dirs)} patient directories")

        # Create paired file mapping
        self.file_pairs = self._create_file_pairs(self.patient_dirs)
        print(f"Created {len(self.file_pairs)} HSI-OCTA file pairs")

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

    def _find_patient_dirs(self, parent_dir: Path) -> list:
        """
        Recursively find all directories that contain patient data.

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
            hsi_files = list(dir_path.glob('*.h5'))
            octa_files = list(dir_path.glob('*RetinaAngiographyEnface*.tiff'))

            if hsi_files and octa_files:
                patient_dirs.append(dir_path)

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
            hsi_file = None
            if len(c1_files) == 1:
                hsi_file = c1_files[0]
            elif len(d1_files) == 1:
                hsi_file = d1_files[0]
            elif len(c1_files) > 1:
                print(f"Warning: Multiple C1 files in {patient_dir}, using first one")
                hsi_file = c1_files[0]
            elif len(d1_files) > 1:
                print(f"Warning: Multiple D1 files in {patient_dir}, using first one")
                hsi_file = d1_files[0]

            # Create pair if both HSI and OCTA files are available
            if hsi_file and len(octa_files) == 1:
                # Extract the patient ID from the directory path
                # Use the last part of the path as the patient ID if it doesn't have a specific pattern
                patient_id = patient_dir.name

                pairs.append({
                    'hsi': hsi_file,
                    'octa': octa_files[0],
                    'patient_id': patient_id
                })
            else:
                if not hsi_file:
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
        with h5py.File(hsi_path, 'r') as hsi_file:
            hsi_img = hsi_file['Cube/Images'][:]

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

    def __init__(self, spectral_channels: int = 31):
        """
        Initialize the generator network.

        Args:
            spectral_channels: Number of spectral channels in the input HSI data
        """
        super(Generator, self).__init__()

        # Encoder layers - process 3D input data with spectral dimension reduction
        self.encoder = nn.ModuleList([
            # [B, 1, 31, 500, 500] -> [B, 32, 16, 500, 500]
            # First encoder layer reduces spectral dimensions by half
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),

            # [B, 32, 16, 500, 500] -> [B, 64, 8, 500, 500]
            # Second encoder layer further reduces spectral dimensions
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),

            # [B, 64, 8, 500, 500] -> [B, 128, 4, 500, 500]
            # Third encoder layer creates deep features with reduced spectral dimensions
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        ])

        # Skip connections to preserve spatial information
        self.skip_connections = nn.ModuleList([
            # Process features from first encoder layer for skip connection
            nn.Sequential(
                nn.Conv3d(32, 32, 1),  # 1x1x1 convolution to refine features
                nn.BatchNorm3d(32),
                nn.ReLU()
            ),
            # Process features from second encoder layer for skip connection
            nn.Sequential(
                nn.Conv3d(64, 64, 1),  # 1x1x1 convolution to refine features
                nn.BatchNorm3d(64),
                nn.ReLU()
            )
        ])

        # Channel reduction layers after concatenation with skip connections
        self.reduction_layers = nn.ModuleList([
            # Reduce channels after first skip concatenation
            nn.Sequential(
                nn.Conv2d(192, 128, 1),  # 128 + 64 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            # Reduce channels after second skip concatenation
            nn.Sequential(
                nn.Conv2d(160, 128, 1),  # 128 + 32 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        ])

        # Decoder layers with corrected channel dimensions
        self.decoder = nn.ModuleList([
            # First decoder block - maintain high-level features
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),  # 128 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),  # 128 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            # Second decoder block - maintain high-level features
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),  # 128 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),  # 128 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            # Third decoder block - start reducing features
            nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),  # 128 -> 64
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, 1, 1),  # 64 -> 32
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            # Final output layer - generate single-channel OCTA
            nn.Sequential(
                nn.Conv2d(32, 1, 3, 1, 1),  # 32 -> 1
                nn.Tanh()  # Output in range [-1, 1]
            )
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            x: Input HSI tensor with shape [B, 31, 500, 500]

        Returns:
            Generated OCTA tensor with shape [B, 1, 500, 500]
        """
        # Input: [B, 31, 500, 500]
        skips = []
        x = x.unsqueeze(1)  # [B, 1, 31, 500, 500] - Add channel dimension for 3D conv

        # Encoder path with skip connections
        for i, enc_layer in enumerate(self.encoder):
            x = F.relu(enc_layer(x))
            if i < len(self.skip_connections):
                skip = self.skip_connections[i](x)
                skips.append(skip)

        # Collapse spectral dimension with max pooling
        x = x.max(dim=2)[0]  # [B, 128, 500, 500] - Spectral dimension collapsed

        # Decoder path with skip connections
        for i, dec_layer in enumerate(self.decoder[:-1]):
            if i < len(skips):
                # Incorporate skip connection
                skip = skips[-(i + 1)].max(dim=2)[0]  # Collapse spectral dimension
                x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
                x = self.reduction_layers[i](x)  # Reduce channels
            x = dec_layer(x)

        # Final output layer
        return self.decoder[-1](x)


# =====================
# Discriminator Architecture
# =====================
class Discriminator(nn.Module):
    """
    PatchGAN discriminator for differentiating between real and generated OCTA images.

    This discriminator classifies overlapping image patches rather than the whole image,
    allowing it to focus on local texture details rather than global structure.
    """

    def __init__(self):
        """Initialize the PatchGAN discriminator network."""
        super(Discriminator, self).__init__()

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
            *discriminator_block(1, 64, normalize=False),  # No normalization on first layer
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()  # Output in range [0, 1] - probability that patch is real
        )

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


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler for training.

    Reduces learning rate according to schedule to fine-tune models in later stages.

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration dictionary

    Returns:
        Learning rate scheduler
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_scheduler']['lr_decay_interval'],
        gamma=config['lr_scheduler']['lr_decay_factor'],
        last_epoch=-1  # Start fresh from epoch 0
    )


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