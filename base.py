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
    def __init__(self,
                 data_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 augment: bool = True,
                 split: str = 'train',
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42,
                 target_size: int = 500):

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.split = split
        self.target_size = target_size

        # Find all patient directories
        patient_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        # Create paired file mapping
        self.file_pairs = self._create_file_pairs(patient_dirs)

        # Split dataset
        np.random.seed(random_seed)
        indices = np.arange(len(self.file_pairs))
        np.random.shuffle(indices)

        test_size = int(len(indices) * test_ratio)
        val_size = int(len(indices) * val_ratio)
        train_size = len(indices) - test_size - val_size

        if split == 'train':
            self.indices = indices[:train_size]
        elif split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]

        # Define augmentation pipeline
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]) if augment else None

    def _create_file_pairs(self, patient_dirs: List[Path]) -> List[Dict[str, Path]]:
        """Create pairs of corresponding HSI and OCTA files."""
        pairs = []
        for patient_dir in patient_dirs:
            hsi_files = list(patient_dir.glob('*C1*.h5'))
            octa_files = list(patient_dir.glob('*RetinaAngiographyEnface*.tiff'))

            if len(hsi_files) == 1 and len(octa_files) == 1:
                pairs.append({
                    'hsi': hsi_files[0],
                    'octa': octa_files[0],
                    'patient_id': patient_dir.name
                })
            else:
                print(f"Warning: Missing or multiple files in {patient_dir}")

        return pairs

    def _load_hsi(self, hsi_path: Path) -> torch.Tensor:
        """Load and preprocess HSI data, using every third wavelength."""
        with h5py.File(hsi_path, 'r') as hsi_file:
            hsi_img = hsi_file['Cube/Images'][:]
            # Take every third wavelength
            hsi_img = hsi_img[::3]
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

            return hsi_img

    def _load_octa(self, octa_path: Path) -> torch.Tensor:
        """Load and preprocess OCTA image with resizing to target size."""
        octa_img = Image.open(octa_path).convert('L')

        # Resize to target size
        if octa_img.size != (self.target_size, self.target_size):
            octa_img = octa_img.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)

        # Convert to numpy array and normalize to [0,1] range
        octa_img = np.array(octa_img, dtype=np.float32) / 255.0

        # Convert to torch tensor and add channel dimension
        octa_img = torch.tensor(octa_img, dtype=torch.float32).unsqueeze(0)
        return octa_img

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Load and preprocess a pair of HSI and OCTA images."""
        # Get file paths for the requested index
        pair = self.file_pairs[self.indices[idx]]

        # Load both modalities
        hsi_img = self._load_hsi(pair['hsi'])
        octa_img = self._load_octa(pair['octa'])

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
    """Generator network for translating HSI to OCTA images."""

    def __init__(self, spectral_channels: int = 31):
        super(Generator, self).__init__()

        # Encoder layers
        self.encoder = nn.ModuleList([
            # [B, 1, 31, 500, 500] -> [B, 32, 16, 500, 500]
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),

            # [B, 32, 16, 500, 500] -> [B, 64, 8, 500, 500]
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),

            # [B, 64, 8, 500, 500] -> [B, 128, 4, 500, 500]
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        ])

        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(32, 32, 1),
                nn.BatchNorm3d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv3d(64, 64, 1),
                nn.BatchNorm3d(64),
                nn.ReLU()
            )
        ])

        # Channel reduction layers after concatenation
        self.reduction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(192, 128, 1),  # 128 + 64 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(160, 128, 1),  # 128 + 32 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        ])

        # Decoder layers with corrected channel dimensions
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),  # 128 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),  # 128 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),  # 128 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),  # 128 -> 128
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),  # 128 -> 64
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, 1, 1),  # 64 -> 32
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, 3, 1, 1),  # 32 -> 1
                nn.Tanh()
            )
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 31, 500, 500]
        skips = []
        x = x.unsqueeze(1)  # [B, 1, 31, 500, 500]

        # Encoder path
        for i, enc_layer in enumerate(self.encoder):
            x = F.relu(enc_layer(x))
            if i < len(self.skip_connections):
                skip = self.skip_connections[i](x)
                skips.append(skip)

        # Collapse spectral dimension
        x = x.max(dim=2)[0]  # [B, 128, 500, 500]

        # Decoder path with skip connections
        for i, dec_layer in enumerate(self.decoder[:-1]):
            if i < len(skips):
                skip = skips[-(i + 1)].max(dim=2)[0]
                x = torch.cat([x, skip], dim=1)
                x = self.reduction_layers[i](x)
            x = dec_layer(x)

        return self.decoder[-1](x)


# =====================
# Discriminator Architecture
# =====================
class Discriminator(nn.Module):
    """PatchGAN discriminator for differentiating between real and generated OCTA images."""

    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels: int, out_channels: int,
                                normalize: bool = True) -> List[nn.Module]:
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # For 500x500 input, output will be 31x31 patches
        self.model = nn.Sequential(
            *discriminator_block(1, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =====================
# Perceptual Loss
# =====================
class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features."""

    def __init__(self, layers: List[str] = ['3', '8', '15', '22']):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        vgg.eval()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.feature_extractors = nn.ModuleList()
        previous_layer = 0

        for layer_idx in layers:
            layer_idx = int(layer_idx)
            layers = []
            for module in list(vgg.features.children())[previous_layer:layer_idx]:
                if isinstance(module, nn.ReLU):
                    # Replace in-place ReLU with regular ReLU
                    layers.append(nn.ReLU(inplace=False))
                else:
                    layers.append(module)
            sequential = nn.Sequential(*layers)
            self.feature_extractors.append(sequential)
            previous_layer = layer_idx

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert single-channel image to 3 channels and normalize."""
        # Clone input to avoid in-place modifications
        x = x.clone()

        # Repeat the single channel 3 times
        x = x.repeat(1, 3, 1, 1)

        # Normalize with ImageNet statistics
        x = self.normalize(x)
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Preprocess inputs
        x = self._preprocess(x)
        y = self._preprocess(y)

        loss = 0
        for extractor in self.feature_extractors:
            x = extractor(x)
            y = extractor(y)

            # Normalize features safely without in-place operations
            x_norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8
            y_norm = torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8

            x = x / x_norm
            y = y / y_norm

            loss = loss + F.mse_loss(x, y)

        return loss


# =====================
# SSIM Loss
# =====================
class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for preserving structural information.
    """

    def __init__(self, window_size: int = 11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.register_buffer('window', self._create_window())
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _create_window(self) -> torch.Tensor:
        """Creates a Gaussian window for SSIM computation."""
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

        return window.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Clone inputs to avoid in-place modifications
        x = x.clone()
        y = y.clone()

        window = self.window.expand(x.size(1), 1, self.window_size, self.window_size)

        # Compute means
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

        return 1 - ssim_map.mean()
# =====================
# Training Configuration
# =====================
class TrainingConfig:
    """Configuration class for training hyperparameters and settings."""

    def __init__(self):
        # Training hyperparameters
        self.num_epochs = 200
        self.batch_size = 8
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999

        # Loss weights
        self.lambda_pixel = 100.0  # L1 loss weight
        self.lambda_perceptual = 10.0  # Perceptual loss weight
        self.lambda_ssim = 5.0  # SSIM loss weight
        self.lambda_adv = 1.0  # Adversarial loss weight

        # Training settings
        self.save_interval = 10  # Save model every N epochs
        self.print_interval = 100  # Print progress every N batches
        self.validate_interval = 5  # Run validation every N epochs

        # Optimizer settings
        self.weight_decay = 1e-4
        self.gradient_clip = 1.0

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

    Args:
        model (nn.Module): Neural network module to initialize
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(model.weight.data)
        if model.bias is not None:
            init.constant_(model.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0.0)


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler for training.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule
        config (dict): Training configuration dictionary

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_scheduler']['lr_decay_interval'],
        gamma=config['lr_scheduler']['lr_decay_factor'],
        last_epoch=-1
    )


# =====================
# Training Utils
# =====================
def save_checkpoint(state: Dict, filename: str) -> None:
    """
    Save training checkpoint to disk.

    Args:
        state (Dict): Dictionary containing model and training state
        filename (str): Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    """
    Load training checkpoint from disk.

    Args:
        filename (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state into

    Returns:
        int: The epoch number where training left off
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']