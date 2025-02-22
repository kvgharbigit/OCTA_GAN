import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from PIL import Image
import os
from torch.nn import init
from typing import List, Tuple, Optional


# =====================
# Dataset Implementation
# =====================
class HSI_OCTA_Dataset(Dataset):
    """
    Custom Dataset class for paired HSI (Hyperspectral Image) and OCTA (Optical Coherence Tomography Angiography) images.
    Includes data augmentation capabilities for robust training.

    Args:
        data_dir (str): Directory containing paired .h5 (HSI) and .jpeg (OCTA) files
        transform (Optional[transforms.Compose]): Normalization transforms
        augment (bool): Whether to apply data augmentation
    """

    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None,
                 augment: bool = True):
        self.data_dir = data_dir
        # Get all HSI files (.h5 format)
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.transform = transform
        self.augment = augment

        # Define augmentation pipeline with multiple transforms
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Random horizontal flipping
            transforms.RandomVerticalFlip(),  # Random vertical flipping
            transforms.RandomRotation(10),  # Random rotation up to 10 degrees
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a pair of HSI and OCTA images.

        Args:
            idx (int): Index of the image pair to load

        Returns:
            tuple: (HSI tensor, OCTA tensor)
        """
        # Construct file paths for both modalities
        hsi_path = os.path.join(self.data_dir, self.file_list[idx])
        octa_path = hsi_path.replace('.h5', '.jpeg')

        # Load HSI data from HDF5 file
        with h5py.File(hsi_path, 'r') as hsi_file:
            hsi_img = hsi_file['hsi'][:]

        # Load and normalize OCTA image to [0,1] range
        octa_img = Image.open(octa_path).convert('L')
        octa_img = np.array(octa_img, dtype=np.float32) / 255.0

        # Convert numpy arrays to PyTorch tensors with correct dimensions
        hsi_img = torch.tensor(hsi_img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        octa_img = torch.tensor(octa_img, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        # Apply same augmentation to both images to maintain correspondence
        if self.augment and self.aug_transforms:
            combined = torch.cat([hsi_img, octa_img], dim=0)
            combined = self.aug_transforms(combined)
            hsi_img = combined[:hsi_img.shape[0]]
            octa_img = combined[-1:]

            # Apply normalization transforms
        if self.transform:
            hsi_img = self.transform(hsi_img)
            octa_img = self.transform(octa_img)

        return hsi_img, octa_img


# =====================
# Self-Attention Mechanism
# =====================
class SelfAttention(nn.Module):
    """
    Self-attention module for capturing long-range dependencies in feature maps.
    Implements the self-attention mechanism from 'Self-Attention Generative Adversarial Networks'.

    Args:
        in_channels (int): Number of input channels
    """

    def __init__(self, in_channels: int):
        super(SelfAttention, self).__init__()
        # Transform input features into query, key, and value
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)  # Query projection
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)  # Key projection
        self.value = nn.Conv2d(in_channels, in_channels, 1)  # Value projection
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute self-attention on input feature maps.

        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]

        Returns:
            torch.Tensor: Self-attended feature map
        """
        batch_size, C, H, W = x.size()

        # Project input into query, key, value spaces
        query = self.query(x).view(batch_size, -1, H * W)  # [B, C/8, N]
        key = self.key(x).view(batch_size, -1, H * W)  # [B, C/8, N]
        value = self.value(x).view(batch_size, -1, H * W)  # [B, C, N]

        # Compute attention scores using scaled dot-product attention
        attention = F.softmax(torch.bmm(query.permute(0, 2, 1), key), dim=-1)  # [B, N, N]

        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch_size, C, H, W)  # [B, C, H, W]

        # Apply learnable scaling and residual connection
        return self.gamma * out + x


# =====================
# Generator Architecture
# =====================
class Generator(nn.Module):
    """
    Generator network for translating HSI to OCTA images.
    Uses a 3D encoder for spectral feature extraction followed by a 2D decoder with skip connections
    and attention mechanisms for spatial reconstruction.

    Args:
        spectral_channels (int): Number of spectral channels in input HSI
    """

    def __init__(self, spectral_channels: int = 100):
        super(Generator, self).__init__()

        # 3D Convolutional encoder for spectral-spatial feature extraction
        self.encoder = nn.ModuleList([
            nn.Conv3d(1, 32, kernel_size=(3, 3, 5), stride=(1, 1, 2), padding=(1, 1, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1))
        ])

        # Skip connections for preserving fine spatial details
        self.skip_connections = nn.ModuleList([
            nn.Conv3d(32, 32, 1),  # 1x1x1 convolution for channel adjustment
            nn.Conv3d(64, 64, 1)
        ])

        # 2D Decoder with attention for spatial reconstruction
        self.decoder = nn.ModuleList([
            # First decoder block with attention
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                SelfAttention(128)
            ),
            # Second decoder block with attention
            nn.Sequential(
                nn.Conv2d(128 + 64, 64, 3, 1, 1),  # +64 for skip connection
                nn.BatchNorm2d(64),
                nn.ReLU(),
                SelfAttention(64)
            ),
            # Third decoder block
            nn.Sequential(
                nn.Conv2d(64 + 32, 32, 3, 1, 1),  # +32 for skip connection
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            # Final output layer
            nn.Sequential(
                nn.Conv2d(32, 1, 3, 1, 1),
                nn.Tanh()  # Output in range [-1, 1]
            )
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            x (torch.Tensor): Input HSI tensor [B, C, H, W]

        Returns:
            torch.Tensor: Generated OCTA image [B, 1, H, W]
        """
        skips = []  # Store skip connections

        # Encoder pathway
        x = x.unsqueeze(1)  # Add channel dimension for 3D convolution
        for i, enc_layer in enumerate(self.encoder):
            x = F.relu(enc_layer(x))
            if i < len(self.skip_connections):
                skips.append(self.skip_connections[i](x))

        # Remove spectral dimension before 2D decoding
        x = x.squeeze(-1)

        # Decoder pathway with skip connections
        for i, dec_layer in enumerate(self.decoder[:-1]):
            if i < len(skips):
                skip = skips[-(i + 1)].squeeze(-1)
                x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
            x = dec_layer(x)

        return self.decoder[-1](x)


# =====================
# Discriminator Architecture
# =====================
class Discriminator(nn.Module):
    """
    PatchGAN discriminator for differentiating between real and generated OCTA images.
    Uses a series of strided convolutions to classify patches of the input image.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels: int, out_channels: int,
                                normalize: bool = True) -> List[nn.Module]:
            """Helper function to create discriminator blocks"""
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Build discriminator architecture
        self.model = nn.Sequential(
            *discriminator_block(1, 64, normalize=False),  # First layer without normalization
            *discriminator_block(64, 128),  # Intermediate layers
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),  # Output layer
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input image [B, 1, H, W]

        Returns:
            torch.Tensor: Probability map of real vs. fake
        """
        return self.model(x)


# =====================
# Perceptual Loss
# =====================
class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    Compares images in feature space rather than pixel space.

    Args:
        layers (List[str]): VGG16 layer indices to extract features from
    """

    def __init__(self, layers: List[str] = ['3', '8', '15', '22']):
        super(PerceptualLoss, self).__init__()

        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)
        vgg.eval()

        # Convert input from grayscale to RGB-like features
        self.first_layer = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Create feature extractors for each layer
        self.feature_extractors = nn.ModuleList()
        previous_layer = 0

        for layer_idx in layers:
            layer_idx = int(layer_idx)
            sequential = nn.Sequential(
                *list(vgg.features.children())[previous_layer:layer_idx]
            )
            self.feature_extractors.append(sequential)
            previous_layer = layer_idx

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between two images.

        Args:
            x (torch.Tensor): Generated image
            y (torch.Tensor): Target image

        Returns:
            torch.Tensor: Perceptual loss value
        """
        # Convert grayscale to RGB-like features
        x = self.first_layer(x)
        y = self.first_layer(y)

        loss = 0
        # Compare features at each selected layer
        for extractor in self.feature_extractors:
            x = extractor(x)
            y = extractor(y)
            # Normalize features
            x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
            y = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8)
            loss += F.mse_loss(x, y)

        return loss


# =====================
# SSIM Loss
# =====================
class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for preserving structural information.
    Computes SSIM between generated and target images.

    Args:
        window_size (int): Size of the sliding window for SSIM computation
    """

    def __init__(self, window_size: int = 11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss between two images.

        Args:
            x (torch.Tensor): Generated image
            y (torch.Tensor): Target image

        Returns:
            torch.Tensor: 1 - SSIM (loss value)
        """
        # Constants for numerical stability
        C1 = (0.01 * 2) ** 2
        C2 = (0.03 * 2) ** 2

        # Compute means
        mu_x = F.avg_pool2d(x, self.window_size, stride=1)