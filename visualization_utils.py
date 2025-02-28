import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def select_representative_wavelengths(total_wavelengths=31, min_wavelength=450, max_wavelength=905):
    """
    Select representative wavelengths across the spectral range,
    considering every third wavelength from the original 91 wavelengths.

    Args:
        total_wavelengths (int): Total number of wavelengths (31 from every 3rd of 91)
        min_wavelength (int): Minimum wavelength in nm
        max_wavelength (int): Maximum wavelength in nm

    Returns:
        dict: Mapping of color channels to wavelength indices
    """
    # Generate wavelength array for the 31 wavelengths
    # Mapping back to the original 91 wavelengths range
    full_wavelengths = np.linspace(min_wavelength, max_wavelength, 91)
    sampled_wavelengths = full_wavelengths[::3][:total_wavelengths]

    # Define key wavelengths for human vision
    wavelength_map = {
        'red': 660,  # Deep red
        'green': 555,  # Peak green perception
        'blue': 475  # Blue-green border
    }

    # Find indices closest to these wavelengths in the sampled wavelengths
    wavelength_indices = {}
    for color, target_wavelength in wavelength_map.items():
        wavelength_indices[color] = np.argmin(np.abs(sampled_wavelengths - target_wavelength))

    return wavelength_indices


def normalize_wavelength_image(wavelength_image):
    """
    Normalize an image for display, handling potential outliers.

    Args:
        wavelength_image (torch.Tensor or np.ndarray): Single wavelength image

    Returns:
        torch.Tensor: Normalized image
    """
    import numpy as np
    import torch

    # Convert to numpy if it's a tensor
    if torch.is_tensor(wavelength_image):
        img_np = wavelength_image.squeeze().cpu().numpy()
    else:
        img_np = np.squeeze(wavelength_image)

    # Clip to remove extreme outliers (99th percentile)
    lower = np.percentile(img_np, 1)
    upper = np.percentile(img_np, 99)

    # Clip and normalize
    img_np = np.clip(img_np, lower, upper)

    # Min-max normalization
    if upper > lower:
        img_np = (img_np - lower) / (upper - lower)

    return torch.from_numpy(img_np).unsqueeze(0)


def save_sample_visualizations(generator, val_loader, device, writer, epoch, output_dir):
    """
    Generate and save a single sample visualization during training.

    Args:
        generator (nn.Module): The generator model
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to run the model on
        writer (SummaryWriter): TensorBoard writer
        epoch (int): Current training epoch
        output_dir (Path): Directory to save visualizations
    """
    generator.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Take the first sample from the validation loader
        hsi, octa, patient_ids = next(iter(val_loader))

        # Move data to device
        hsi = hsi.to(device)
        octa = octa.to(device)

        # Take just the first sample in the batch
        hsi = hsi[0:1]
        octa = octa[0:1]

        # Generate fake OCTA image
        fake_octa = generator(hsi)

        # Denormalize images to [0, 1] range for visualization
        hsi_norm = (hsi + 1) / 2  # Assuming normalization was [-1, 1]
        octa_norm = (octa + 1) / 2
        fake_octa_norm = (fake_octa + 1) / 2

        # Select representative wavelength indices
        wavelength_indices = select_representative_wavelengths()

        # Create RGB-like representation of HSI
        hsi_np = hsi_norm.squeeze().cpu().numpy()

        # Select wavelength indices for RGB-like representation
        rgb_hsi = np.stack([
            hsi_np[wavelength_indices['red']],  # Red channel
            hsi_np[wavelength_indices['green']],  # Green channel
            hsi_np[wavelength_indices['blue']]  # Blue channel
        ])

        # Normalize each channel
        rgb_hsi = np.stack([
            normalize_wavelength_image(rgb_hsi[0]).squeeze(),
            normalize_wavelength_image(rgb_hsi[1]).squeeze(),
            normalize_wavelength_image(rgb_hsi[2]).squeeze()
        ])

        # Ensure HSI is a 3-channel image
        hsi_rgb = np.transpose(rgb_hsi, (1, 2, 0))  # Convert to HWC

        # Convert HSI to tensor with 3 channels
        hsi_rgb_tensor = torch.from_numpy(hsi_rgb).permute(2, 0, 1)

        # Convert OCTA images to grayscale images (3 channel)
        fake_octa_gray = fake_octa_norm.squeeze().repeat(3, 1, 1)
        octa_gray = octa_norm.squeeze().repeat(3, 1, 1)

        # Create a grid of images: HSI (RGB-like), Generated OCTA, Real OCTA
        img_grid = torch.stack([
            hsi_rgb_tensor,  # HSI (RGB-like)
            fake_octa_gray,  # Generated OCTA
            octa_gray  # Real OCTA
        ])

        # Log to TensorBoard
        writer.add_images(f'Samples/epoch_{epoch}', img_grid, 0)

        # Save as image file
        save_image_grid(
            img_grid,
            output_dir / f'epoch_{epoch}_sample.png'
        )


def save_image_grid(img_tensor, filepath):
    """
    Save a grid of images to a file.

    Args:
        img_tensor (torch.Tensor): Image tensor, shape [num_images, C, H, W]
        filepath (Path): Path to save the image
    """
    # Create grid of images
    grid = vutils.make_grid(img_tensor, normalize=True, scale_each=True)

    # Convert to numpy and plot
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()